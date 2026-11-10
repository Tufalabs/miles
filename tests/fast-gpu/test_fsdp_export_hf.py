"""An FSDP2 HF export reloads with transformers and matches the training weights."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=60, suite="stage-b-2-gpu-h200", labels=["fsdp"], hardware=["hopper", "blackwell"], num_gpus=2
)

import os
import subprocess
import sys
from pathlib import Path

import pytest

_WORKER = Path(__file__).with_name("_fsdp_export_hf_worker.py")
_REPO_ROOT = Path(__file__).parents[2]


def test_fsdp_export_hf_reloads_with_transformers(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            str(_WORKER),
            "--base-dir",
            str(tmp_path / "base"),
            "--export-dir",
            str(tmp_path / "export"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS fsdp export_hf" in result.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
