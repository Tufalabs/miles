"""Two-rank worker: export an FSDP2-sharded tiny Qwen3 as an HF checkpoint and reload it."""

import argparse
import json
import os
from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM

from miles.backends.fsdp_utils.actor import apply_fsdp2
from miles.backends.fsdp_utils.hf_weight_iterator import FSDPHfWeightIterator
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.distributed_utils import init_gloo_group
from miles.utils.hf_utils.config import is_complete_hf_export


def _tiny_qwen3_config() -> Qwen3Config:
    return Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
        tie_word_embeddings=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    args = parser.parse_args()
    base_dir, export_dir = Path(args.base_dir), Path(args.export_dir)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    init_gloo_group()
    rank = dist.get_rank()

    if rank == 0:
        torch.manual_seed(1234)
        Qwen3ForCausalLM(_tiny_qwen3_config()).save_pretrained(base_dir)
    dist.barrier()
    model = Qwen3ForCausalLM.from_pretrained(base_dir).cuda()
    reference = {name: tensor.detach().clone().cpu() for name, tensor in model.state_dict().items()}
    apply_fsdp2(model, args=Namespace(backend=Namespace(fp16=False)))

    iterator = FSDPHfWeightIterator.build(
        Namespace(update_weight_buffer_size=16 << 10),  # a few KB per shard, so the export spans many shards
        model,
        required_placement=WeightUpdatePlacement(gather_pp=True),
        model_name="qwen3forcausallm",
        quantization_config=None,
    )
    SnapshotPublisher(iterator).publish_model(export_dir, hf_checkpoint=str(base_dir))

    if rank == 0:
        assert is_complete_hf_export(export_dir), f"no completion marker in {os.listdir(export_dir)}"
        index = json.loads((export_dir / "model.safetensors.index.json").read_text())
        assert len(set(index["weight_map"].values())) > 1, index
        assert (export_dir / "config.json").exists()
        exported = AutoModelForCausalLM.from_pretrained(export_dir).state_dict()
        assert exported.keys() == reference.keys(), (sorted(exported), sorted(reference))
        for name, expected in reference.items():
            assert exported[name].dtype == expected.dtype, name
            assert torch.equal(exported[name], expected), name
        print("PASS fsdp export_hf", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
