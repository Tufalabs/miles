"""A trainer-side adapter that tracks the sampler (experiment, LLMT-83).

MILES_SAMPLER_ADAPTER names a group of parameters (entries as MILES_FREEZE_PARAMS: substrings of parameter
names, and `qkv_rows:<letters of q, k, v>`). The group then has two sets of values. The model's own are the
ones the sampler has: the optimizer, the weight sync and the checkpoints see them. The trainer adds an offset
to them for its forward passes. Each training step:

  1. the offset is added to the group;
  2. a pass whose only loss is the divergence between trainer and sampler over the sampler's returned
     candidates (loss/hub/score_centering_head_kl.py) gives the offset's gradient;
  3. the policy pass runs as usual, through the adapted weights; its gradient goes to the model's own values
     as for any other weight (MILES_SAMPLER_ADAPTER_FREEZE=1 zeroes it for the group, which then stays where
     the run started on the sampler's side);
  4. the group goes back to the sampler's values and the offset takes its step, offset <- (1 - lr * wd) *
     offset - lr * gradient (MILES_SAMPLER_ADAPTER_LR, default 0.02; MILES_SAMPLER_ADAPTER_WD, default 0).
     The weight decay gives the offset a memory of 1 / (lr * wd) steps: it holds what the last steps'
     divergence supports and forgets the rest, so that it does not accumulate a change of its own.

So the sampler never receives the offset and the policy loss never trains it. The offset is kept in fp32; the
model's bf16 values during the passes are the rounded sum of the sampler's bf16 values and the offset. It
doubles the training passes of a step.

The offset is saved next to each checkpoint, one file per rank
(<checkpoints>/sampler_adapter_iter_<step>/rank<rank>.pt), and a run that loads that checkpoint starts from it.
A checkpoint without such files starts the offset from zero, or from MILES_SAMPLER_ADAPTER_INIT=<file>: one
such file, read by every rank (the group must be one whose values the ranks share, as attention's are under
expert parallelism). MILES_SAMPLER_ADAPTER_INIT_OVER_CHECKPOINT=1 uses that file even when the checkpoint has
offsets of its own.

MILES_SAMPLER_ADAPTER_LR=0 holds the offset where it starts: no divergence pass, one training pass a step. With
MILES_SAMPLER_ADAPTER_INIT that is a fixed correction of the trainer's forward, fitted elsewhere.

MILES_SAMPLER_ADAPTER_EVERY=<n> (default 1) takes the divergence pass and the offset's step only on the rollouts
whose id is a multiple of n; the other steps hold the offset and take one training pass.

MILES_SAMPLER_ADAPTER_RANK=<r> keeps the offset of every matrix in the group at rank r: after each step the
offset is replaced by its rank-r approximation (`_truncate`). The q, k and v rows of a linear_qkv weight are
three matrices, each at rank r. Vectors (norm gains) are left as they are. A low-rank offset holds less of
one batch's noise per step, which is what a larger MILES_SAMPLER_ADAPTER_LR needs.

MILES_SAMPLER_ADAPTER_INPUT_STATS=1 logs per attention layer, over the tokens of the step's adapted passes, how
much of what the offset adds to q, k and v is the same for every token (`_InputStats`). For a measurement at
learning rate 0; it needs the whole of each linear_qkv.weight in the group.
"""

import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from miles.backends.training_utils.loss.hub import score_centering_head_kl

from .weight_swap import _qkv_row_mask

logger = logging.getLogger(__name__)

ENV_VAR = "MILES_SAMPLER_ADAPTER"
LR_ENV_VAR = "MILES_SAMPLER_ADAPTER_LR"
_ROWS_PREFIX = "qkv_rows:"
_ENTRIES = tuple(p for p in os.environ.get(ENV_VAR, "").split(",") if p)
_PATTERNS = tuple(p for p in _ENTRIES if not p.startswith(_ROWS_PREFIX))
_QKV_PARTS = "".join(p[len(_ROWS_PREFIX) :] for p in _ENTRIES if p.startswith(_ROWS_PREFIX))
_LR = float(os.environ.get(LR_ENV_VAR, "") or 0.02)
WD_ENV_VAR = "MILES_SAMPLER_ADAPTER_WD"
_WD = float(os.environ.get(WD_ENV_VAR, "") or 0.0)
assert 0 <= _LR * _WD < 1, f"{WD_ENV_VAR}={_WD} with {LR_ENV_VAR}={_LR}: lr * wd must be in [0, 1)"
EVERY_ENV_VAR = "MILES_SAMPLER_ADAPTER_EVERY"
_EVERY = int(os.environ.get(EVERY_ENV_VAR, "") or 1)
assert _EVERY >= 1, f"{EVERY_ENV_VAR}={_EVERY}: expected a positive number of steps"
RANK_ENV_VAR = "MILES_SAMPLER_ADAPTER_RANK"
_RANK = int(os.environ.get(RANK_ENV_VAR, "") or 0)
assert _RANK >= 0, f"{RANK_ENV_VAR}={_RANK}: expected a rank, or 0 for no limit"
_SUBSPACE_ROUNDS = 2
INIT_ENV_VAR = "MILES_SAMPLER_ADAPTER_INIT"
_INIT = os.environ.get(INIT_ENV_VAR, "")
_INIT_OVER_CHECKPOINT = os.environ.get(f"{INIT_ENV_VAR}_OVER_CHECKPOINT", "") == "1"
assert _INIT or not _INIT_OVER_CHECKPOINT, f"{INIT_ENV_VAR}_OVER_CHECKPOINT=1 needs {INIT_ENV_VAR}"
assert _LR > 0 or _INIT, f"{LR_ENV_VAR}=0 holds the offset at zero without {INIT_ENV_VAR}"
FREEZE_ENV_VAR = "MILES_SAMPLER_ADAPTER_FREEZE"
_FREEZE = os.environ.get(FREEZE_ENV_VAR, "") == "1"
INPUT_STATS_ENV_VAR = "MILES_SAMPLER_ADAPTER_INPUT_STATS"
_INPUT_STATS = os.environ.get(INPUT_STATS_ENV_VAR, "") == "1"
_INPUT_STATS_DIMS = tuple(int(d) for d in os.environ.get(f"{INPUT_STATS_ENV_VAR}_DIMS", "940").split(","))

# Per adapted parameter: the parameter, the index of its adapted rows (a slice for all of it), the sampler's
# values of those rows, the offset (fp32) and the step's gradient of the divergence on them.
_state: list[dict] = []
_ready = False
# The checkpoint the run loaded (its directory and step), whose saved offsets the first step restores.
_loaded_checkpoint: tuple[str, int] | None = None
_input_stats: list["_InputStats"] = []
# Position buckets (first index, one past the last) for the per-dimension RMS of the input statistics.
_POSITIONS = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 64), (64, 256), (256, 10**9))


class _InputStats:
    """What one layer's linear_qkv receives and what the offset makes of it, summed over tokens.

    x is the layer's normalized input (RMSNorm with the layer's gain, as linear_qkv applies it before its
    matrix). For x itself, for W x (the sampler's weight) and for D x (the offset), the share of the mean
    square that the mean over tokens holds: |mean|^2 / mean |.|^2, per part (q, k, v rows). A share near 1
    says the term is a constant added to every token: a bias. Also the mean and RMS of single dimensions of x,
    and where their energy sits: the share of the tokens that holds 90% of the dimension's sum of squares
    (per pass, averaged) and its RMS by position (`_POSITIONS`): the token's position in its sequence when the
    batch is packed (`positions=sequence` in the line), along the input's first axis otherwise (`positions=row`).
    The last sequence of a packed row is its padding (training_utils/data/rollout.py; a real sequence only when
    the row needed none): it is left out of both and its RMS is given apart (`x<dim>_last_seq_rms`).
    """

    def __init__(self, name: str, attention: torch.nn.Module, entry: dict) -> None:
        self.name, self.entry = name, entry
        weight = attention.linear_qkv.weight
        self.gain = attention.linear_qkv.layer_norm_weight
        self.parts = {part: _qkv_row_mask(attention, weight, part) for part in "qkv"}
        self.eps = float(getattr(attention.linear_qkv, "eps", 1e-6))
        self.bounds: tuple[int, ...] | None = None
        self.reset()
        attention.register_forward_pre_hook(self._capture_bounds, with_kwargs=True)
        attention.linear_qkv.register_forward_pre_hook(self._hook)

    def reset(self) -> None:
        self.tokens = 0
        self.sums: dict[str, torch.Tensor] = {}

    def _add(self, key: str, value: torch.Tensor) -> None:
        value = value.double()
        self.sums[key] = value if key not in self.sums else self.sums[key] + value

    def _capture_bounds(self, module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        """The sequence bounds of a packed batch, for the linear_qkv call that follows."""
        self.bounds = getattr(kwargs.get("packed_seq_params"), "cu_seqlens_host", None)

    @torch.no_grad()
    def _hook(self, module: torch.nn.Module, inputs: tuple) -> None:
        if "sampler" not in self.entry:  # only passes through the adapted weights
            return
        hidden = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
        matrices = (("w", self.entry["sampler"].float()), ("d", self.entry["offset"]))
        picked = []
        for chunk in hidden.split(8192):
            chunk = chunk.float()
            x = chunk * torch.rsqrt(chunk.square().mean(-1, keepdim=True) + self.eps) * self.gain.detach().float()
            picked.append(x[:, list(_INPUT_STATS_DIMS)])
            self.tokens += x.shape[0]
            self._add("x", x.sum(0))
            self._add("x2", x.square().sum(0))
            for label, matrix in matrices:
                y = x @ matrix.T
                self._add(label, y.sum(0))
                self._add(f"{label}2", y.square().sum(0))
        self._add_dims(torch.cat(picked).square(), inputs[0].shape)

    def _add_dims(self, squares: torch.Tensor, shape: torch.Size) -> None:
        """`squares`: per token (rows, in the input's order) the square of x at each of the picked dimensions."""
        self.shape = tuple(shape)
        if self.bounds is not None:
            self.positions = "sequence"
            padding = squares[self.bounds[-2] : self.bounds[-1]]
            self._add("last2", padding.sum(0))
            self._add("lastn", torch.tensor(float(padding.shape[0])))
            squares = squares[: self.bounds[-2]]
            position = torch.empty(squares.shape[0], dtype=torch.long, device=squares.device)
            for start, end in zip(self.bounds[:-2], self.bounds[1:-1], strict=True):
                position[start:end] = torch.arange(end - start, device=squares.device)
        elif len(shape) == 3:
            self.positions = "row"
            position = torch.arange(shape[0], device=squares.device).repeat_interleave(shape[1])
        else:
            self.positions = "none"
            position = torch.full((squares.shape[0],), -1, device=squares.device)
        ordered = squares.sort(0, descending=True).values.cumsum(0)
        tokens90 = (ordered < 0.9 * ordered[-1]).sum(0) + 1
        self._add("tokens90", tokens90 / squares.shape[0])
        self._add("passes", torch.ones(()))
        for first, end in _POSITIONS:
            inside = (position >= first) & (position < end)
            self._add(f"pos2 {first}", squares[inside].sum(0))
            self._add(f"posn {first}", inside.sum())

    def line(self) -> str:
        n = self.tokens
        out = [f"layer={self.name} tokens={n}"]
        mean, mean2 = self.sums["x"] / n, self.sums["x2"] / n
        out.append(f"x_const={float(mean.square().sum() / mean2.sum()):.4f}")
        for dim in _INPUT_STATS_DIMS:
            share = float(mean[dim].square() / mean.square().sum())
            out.append(f"x{dim}_mean={float(mean[dim]):.4f} x{dim}_rms={float(mean2[dim].sqrt()):.4f} x{dim}_of_mean={share:.4f}")
        out.append(f"shape={'x'.join(str(s) for s in self.shape)} positions={self.positions}")
        for i, dim in enumerate(_INPUT_STATS_DIMS):
            out.append(f"x{dim}_tokens90={float(self.sums['tokens90'][i] / self.sums['passes']):.5f}")
            rms = [(self.sums[f"pos2 {first}"][i] / self.sums[f"posn {first}"].clamp_min(1)).sqrt() for first, _ in _POSITIONS if f"pos2 {first}" in self.sums]
            out.append(f"x{dim}_pos_rms=" + "/".join(f"{float(v):.3f}" for v in rms))
            if "last2" in self.sums:
                out.append(f"x{dim}_last_seq_rms={float((self.sums['last2'][i] / self.sums['lastn'].clamp_min(1)).sqrt()):.3f}")
        for label in ("w", "d"):
            mean, mean2 = self.sums[label] / n, self.sums[f"{label}2"] / n
            for part, rows in self.parts.items():
                out.append(f"{label}_{part}_const={float(mean[rows].square().sum() / mean2[rows].sum()):.4f}")
                out.append(f"{label}_{part}_rms={float(mean2[rows].mean().sqrt()):.4e}")
        return " ".join(out)


def enabled() -> bool:
    return bool(_ENTRIES)


@torch.no_grad()
def _rank_blocks(model_chunks) -> None:
    """Per adapted matrix, the row sets that are kept at rank _RANK: q, k and v of a whole linear_qkv weight, else
    all of its adapted rows."""
    attention_of = {}
    for chunk in model_chunks:
        for _, module in chunk.named_modules():
            linear_qkv = getattr(module, "linear_qkv", None)
            if linear_qkv is not None and hasattr(module, "num_query_groups_per_partition"):
                attention_of[id(linear_qkv.weight)] = module
    for entry in _state:
        offset = entry["offset"]
        if offset.dim() != 2:
            continue
        attention = attention_of.get(id(entry["param"])) if isinstance(entry["rows"], slice) else None
        if attention is None:
            masks = [torch.ones(offset.shape[0], dtype=torch.bool)]
        else:
            masks = [_qkv_row_mask(attention, entry["param"], part) for part in "qkv"]
        entry["blocks"] = [{"rows": mask.to(offset.device), "basis": None} for mask in masks]


@torch.no_grad()
def _truncate(block: dict, offset: torch.Tensor) -> None:
    """Replace the block's rows of the offset by a rank-_RANK approximation: its projection on the leading right
    singular directions, found by subspace iteration that starts from the previous step's directions (from a
    fixed random basis the first time, the same on every rank)."""
    matrix = offset[block["rows"]]
    rank = min(_RANK, *matrix.shape)
    basis = block["basis"]
    if basis is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
        basis = torch.randn(matrix.shape[1], rank, generator=generator).to(matrix)
    for _ in range(_SUBSPACE_ROUNDS):
        left = torch.linalg.qr(matrix @ basis).Q
        basis = torch.linalg.qr(matrix.T @ left).Q
    offset[block["rows"]] = (matrix @ basis) @ basis.T
    block["basis"] = basis


def _grad_of(param: torch.Tensor) -> torch.Tensor:
    grad = getattr(param, "main_grad", None)  # config-access-exempt: DDP attaches main_grad at runtime
    return param.grad if grad is None else grad


@torch.no_grad()
def _collect(model_chunks) -> None:
    global _ready
    _ready = True
    whole: set[int] = set()
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            if any(pattern in name for pattern in _PATTERNS):
                whole.add(id(param))
                _state.append({"name": name, "param": param, "rows": slice(None)})
        if not _QKV_PARTS:
            continue
        for name, module in chunk.named_modules():
            linear_qkv = getattr(module, "linear_qkv", None)
            if linear_qkv is None or not hasattr(module, "num_query_groups_per_partition"):
                continue
            if id(linear_qkv.weight) in whole:
                continue
            rows = _qkv_row_mask(module, linear_qkv.weight, _QKV_PARTS)
            label = f"{name}.linear_qkv.weight[{_QKV_PARTS}]"
            _state.append({"name": label, "param": linear_qkv.weight, "rows": rows})
    assert _state, f"{ENV_VAR}={_ENTRIES} names no parameter"
    elements = 0
    for entry in _state:
        entry["offset"] = torch.zeros_like(entry["param"].data[entry["rows"]], dtype=torch.float32)
        elements += entry["offset"].numel()
    _restore_offsets()
    if _RANK:
        _rank_blocks(model_chunks)
    if _INPUT_STATS:
        by_param = {id(entry["param"]): entry for entry in _state if entry["rows"] == slice(None)}
        for chunk in model_chunks:
            for name, module in chunk.named_modules():
                linear_qkv = getattr(module, "linear_qkv", None)
                if linear_qkv is not None and id(linear_qkv.weight) in by_param:
                    _input_stats.append(_InputStats(name, module, by_param[id(linear_qkv.weight)]))
        assert _input_stats, f"{INPUT_STATS_ENV_VAR} needs whole linear_qkv weights in {ENV_VAR}={_ENTRIES}"
    logger.info(
        "sampler adapter: entries=%s tensors=%d elements=%.4e lr=%g wd=%g freeze=%d every=%d rank=%d (this rank)",
        _ENTRIES,
        len(_state),
        elements,
        _LR,
        _WD,
        int(_FREEZE),
        _EVERY,
        _RANK,
    )


def _offsets_file(checkpoints_dir: str, iteration: int) -> Path:
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    return Path(checkpoints_dir) / f"sampler_adapter_iter_{iteration:07d}" / f"rank{rank:05d}.pt"


def loaded_checkpoint(checkpoints_dir: str, iteration: int) -> None:
    """Called when the run has loaded the checkpoint of `iteration`: the offsets saved with it are restored when
    the group is collected, at the first training step."""
    global _loaded_checkpoint
    _loaded_checkpoint = (checkpoints_dir, iteration)


@torch.no_grad()
def _restore_offsets() -> None:
    path = _offsets_file(*_loaded_checkpoint) if _loaded_checkpoint is not None else None
    if path is None or not path.is_file() or _INIT_OVER_CHECKPOINT:
        if not _INIT:
            logger.info("sampler adapter: no saved offsets at %s, starting from zero (this rank)", path)
            return
        path = Path(_INIT)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    names = [entry["name"] for entry in _state]
    assert saved["names"] == names, f"{path} holds offsets of other parameters than {ENV_VAR}={_ENTRIES} names"
    norm2 = 0.0
    for entry, offset in zip(_state, saved["offsets"], strict=True):
        assert offset.shape == entry["offset"].shape, f"{path}: {entry['name']} has shape {tuple(offset.shape)}"
        entry["offset"].copy_(offset)
        norm2 += float(entry["offset"].square().sum())
    logger.info("sampler adapter: restored offsets from %s, offset_norm=%.4e (this rank)", path, norm2**0.5)


@torch.no_grad()
def save_offsets(checkpoints_dir: str, iteration: int) -> None:
    """Next to the checkpoint of `iteration`. Nothing before the first training step has collected the group."""
    if not _ready:
        return
    path = _offsets_file(checkpoints_dir, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    names = [entry["name"] for entry in _state]
    torch.save({"names": names, "offsets": [entry["offset"].cpu() for entry in _state]}, partial)
    os.replace(partial, path)
    logger.info("sampler adapter: saved offsets to %s (this rank)", path)


@torch.no_grad()
def _write(adapted: bool) -> None:
    """Put the adapted values in the weights, keeping the sampler's (the optimizer's last write) aside, or put
    the sampler's back."""
    for entry in _state:
        if adapted:
            entry["sampler"] = entry["param"].data[entry["rows"]].clone()
            values = (entry["sampler"].float() + entry["offset"]).to(entry["sampler"].dtype)
        else:
            values = entry.pop("sampler")
        entry["param"].data[entry["rows"]] = values


def measure(
    run_pass: Callable[[int], object],
    data_iterator: Sequence,
    model_chunks,
    zero_grads: Callable[[], None],
    num_microbatches: int,
    rollout_id: int = 0,
) -> None:
    """Add the offset to the group and take the gradient of the divergence on it from a pass of its own; leaves
    the gradients zeroed, the data iterators where they were and the offset in the weights for the policy pass.
    On a rollout that MILES_SAMPLER_ADAPTER_EVERY skips, and at learning rate 0, it only adds the offset."""
    if not _ready:
        _collect(model_chunks)
    _write(adapted=True)
    if _LR == 0 or rollout_id % _EVERY != 0:
        return
    offsets = [iterator.offset for iterator in data_iterator]
    with score_centering_head_kl.divergence_phase():
        run_pass(num_microbatches)
    assert score_centering_head_kl.phase_uses() > 0, f"{ENV_VAR} needs the score-centering loss"
    with torch.no_grad():
        for entry in _state:
            entry["gradient"] = _grad_of(entry["param"])[entry["rows"]].float().clone()
    zero_grads()
    for iterator, offset in zip(data_iterator, offsets, strict=True):
        iterator.offset = offset


@torch.no_grad()
def finish(rollout_id: int) -> None:
    """After the policy pass: the sampler's values back in the weights and the offset's step; with
    MILES_SAMPLER_ADAPTER_FREEZE no policy gradient for the group."""
    gradient2, offset2, policy2 = 0.0, 0.0, 0.0
    for entry in _state:
        grad = _grad_of(entry["param"])
        policy2 += float(grad[entry["rows"]].float().square().sum())
        if _FREEZE:
            grad[entry["rows"]] = 0
        gradient = entry.pop("gradient", None)  # None: no divergence pass this step
        if gradient is not None:
            gradient2 += float(gradient.square().sum())
            entry["offset"].mul_(1 - _LR * _WD).add_(gradient, alpha=-_LR)
            for block in entry.get("blocks", ()):
                _truncate(block, entry["offset"])
        offset2 += float(entry["offset"].square().sum())
    _write(adapted=False)
    for stats in _input_stats:
        if stats.tokens:
            logger.info("sampler adapter input stats rollout=%d %s (this rank)", rollout_id, stats.line())
        stats.reset()
    logger.info(
        "sampler adapter rollout=%d divergence_grad_norm=%.4e offset_norm=%.4e group_policy_grad_norm=%.4e (this rank)",
        rollout_id,
        gradient2**0.5,
        offset2**0.5,
        policy2**0.5,
    )
