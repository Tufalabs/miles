"""Generate as usual, then score the same answer again with a prefill pass of the engine (LLMT-83 mismatch probe).

Set as --custom-generate-function-path miles.rollout.generate_hub.prefill_rescore.generate. After the normal
generation, the full sequence (prompt and answer) goes back to the engine as input with one new token, asking for
input log-probs from the first answer token on and their top-k (k = --rollout-top-logprobs-num). The engine
computes the answer tokens in a prefill forward (its prefix match stops where the input log-probs start), so the
same tokens get the decode log-probs of the generation and the prefill log-probs of this pass. Stored in the
sample's metadata: prefill_log_probs [response], prefill_topk_token_ids and prefill_topk_log_probs [response, k]
(ids -1 and log-probs -inf where the engine gave fewer); with --use-rollout-routing-replay also
prefill_routed_experts [len(tokens) - 1, layers, k], the prefill pass's expert choices at the positions
rollout_routed_experts covers. Training is unchanged; the metadata travels with the
rollout data that --dump-details saves.
"""

import numpy as np
import pybase64

from miles.rollout import sglang_rollout
from miles.rollout.sglang_rollout import generate as default_generate
from miles.utils.http_utils import post


async def generate(args, sample, sampling_params, evaluation=False):
    sample = await default_generate(args, sample, sampling_params, evaluation=evaluation)
    response = int(sample.response_length or 0)
    if evaluation or response == 0 or str(sample.status).lower().endswith("aborted"):
        return sample
    start = len(sample.tokens) - response
    k = int(args.rollout_top_logprobs_num or 0)
    params = dict(sampling_params)
    params["max_new_tokens"] = 1
    payload = {
        "input_ids": list(sample.tokens),
        "sampling_params": params,
        "return_logprob": True,
        "logprob_start_len": start,
        "top_logprobs_num": k,
    }
    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True
    headers = sglang_rollout.compute_routing_headers(args, sample) if hasattr(sglang_rollout, "compute_routing_headers") else None
    output = await post(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate", payload, headers=headers)
    meta = output["meta_info"]
    token_logprobs = meta["input_token_logprobs"]
    assert len(token_logprobs) >= response, f"prefill gave {len(token_logprobs)} input log-probs for {response} answer tokens"
    token_logprobs = token_logprobs[-response:] if len(token_logprobs) > response else token_logprobs
    ids = np.full((response, k), -1, dtype=np.int32)
    logps = np.full((response, k), -np.inf, dtype=np.float32)
    top = meta.get("input_top_logprobs") or []
    top = top[-response:] if len(top) > response else top
    for i, entries in enumerate(top):
        for j, entry in enumerate((entries or [])[:k]):
            logps[i, j], ids[i, j] = entry[0], entry[1]
    sample.metadata["prefill_log_probs"] = [float(entry[0]) for entry in token_logprobs]
    sample.metadata["prefill_topk_token_ids"] = ids
    sample.metadata["prefill_topk_log_probs"] = logps
    if args.use_rollout_routing_replay and meta.get("routed_experts"):
        experts = np.frombuffer(pybase64.b64decode(meta["routed_experts"], validate=True), dtype=np.int32)
        rows = len(sample.tokens) - 1
        topk = sample.rollout_routed_experts.shape[-1] if sample.rollout_routed_experts is not None else 8
        assert experts.size >= rows * args.num_layers * topk, f"prefill routing has {experts.size} entries for {rows} positions"
        sample.metadata["prefill_routed_experts"] = experts[: rows * args.num_layers * topk].reshape(rows, args.num_layers, topk).copy()
    return sample
