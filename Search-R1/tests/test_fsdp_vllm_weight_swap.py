import unittest
from unittest.mock import patch

import torch

from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager


class FSDPVLLMWeightSwapTest(unittest.TestCase):
    def test_state_dict_is_not_duplicated_to_cpu_before_sync(self):
        events = []

        class FakeWeight:
            def to(self, device, non_blocking=False):
                raise AssertionError("legacy manager must not duplicate the full state_dict on CPU")

        class FakeModule:
            def state_dict(self):
                events.append("state_dict")
                return {"weight": FakeWeight()}

            def train(self):
                events.append("train")

        class FakeEngine:
            def sync_model_weights(self, params, load_format):
                events.append("sync")

            def offload_model_weights(self):
                events.append("vllm_sleep")

        manager = FSDPVLLMShardingManager.__new__(FSDPVLLMShardingManager)
        manager.module = FakeModule()
        manager.inference_engine = FakeEngine()
        manager.full_params = True
        manager.device_mesh = None
        manager.offload_params = True
        manager.offload_grad = True

        with (
            patch(
                "verl.workers.sharding_manager.fsdp_vllm.offload_fsdp_param_and_grad",
                side_effect=lambda **_: events.append("actor_to_cpu"),
                create=True,
            ),
            patch(
                "verl.workers.sharding_manager.fsdp_vllm.load_fsdp_param_and_grad",
                side_effect=lambda **_: events.append("actor_to_gpu"),
                create=True,
            ),
            patch("verl.workers.sharding_manager.fsdp_vllm.log_gpu_memory_usage"),
            patch("torch.cuda.empty_cache"),
            patch("torch.cuda.current_device", return_value=0),
        ):
            manager.__enter__()
            manager.__exit__(None, None, None)

        self.assertEqual(
            events,
            [
                "state_dict",
                "actor_to_cpu",
                "sync",
                "vllm_sleep",
                "actor_to_gpu",
                "train",
            ],
        )


if __name__ == "__main__":
    unittest.main()
