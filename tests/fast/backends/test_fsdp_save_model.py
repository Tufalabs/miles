from types import SimpleNamespace
from unittest.mock import Mock

from miles.backends.fsdp_utils import actor as actor_module


def _actor(save_hf=None):
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.args = SimpleNamespace(
        debug_rollout_only=False, save_hf=save_hf, backend=SimpleNamespace(save="/tmp/checkpoint", async_save=False)
    )
    actor.role = "actor"
    actor.export_hf = Mock()
    return actor


def test_save_model_delegates_to_checkpoint(monkeypatch):
    actor = _actor()
    save = Mock()
    monkeypatch.setattr(actor_module.checkpoint, "save", save)

    actor.save_model(7)

    save.assert_called_once_with(actor, 7)
    actor.export_hf.assert_not_called()


def test_save_model_exports_hf_at_the_save_hf_path(monkeypatch):
    actor = _actor(save_hf="/tmp/hf/step_{rollout_id}")
    monkeypatch.setattr(actor_module.checkpoint, "save", Mock())

    actor.save_model(7)

    actor.export_hf.assert_called_once_with(7, "/tmp/hf/step_7")
