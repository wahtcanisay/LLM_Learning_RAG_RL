"""Contract tests for the local veRL-to-vLLM compatibility boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import ANY, patch

import torch
from omegaconf import OmegaConf


class TestVLLM027Compatibility(unittest.TestCase):
    def test_constructor_forwards_legacy_scheduler_memory_limits(self):
        from verl.third_party.vllm.vllm_v_0_27_1 import llm as adapter_module

        tokenizer = SimpleNamespace(
            pad_token_id=0,
            name_or_path="/tmp/model",
        )
        model_config = SimpleNamespace(_name_or_path="/tmp/model")

        with patch.object(adapter_module, "_VLLM") as engine_cls:
            adapter_module.LLM(
                actor_module=None,
                tokenizer=tokenizer,
                model_hf_config=model_config,
                max_model_len=896,
                max_num_batched_tokens=1792,
                max_num_seqs=2,
                kv_cache_memory_bytes=134217728,
            )

        engine_cls.assert_called_once_with(
            model="/tmp/model",
            tokenizer="/tmp/model",
            tensor_parallel_size=1,
            dtype="auto",
            enforce_eager=False,
            gpu_memory_utilization=0.5,
            max_model_len=896,
            max_num_batched_tokens=1792,
            max_num_seqs=2,
            kv_cache_memory_bytes=134217728,
            enable_sleep_mode=True,
        )

    def test_rollout_passes_scheduler_memory_limits_to_adapter(self):
        from verl.workers.rollout.vllm_rollout import vllm_rollout

        config = OmegaConf.create(
            {
                "tensor_model_parallel_size": 1,
                "enforce_eager": True,
                "free_cache_engine": True,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.25,
                "prompt_length": 768,
                "response_length": 128,
                "load_format": "dummy_hf",
                "max_num_batched_tokens": 1792,
                "max_num_seqs": 2,
                "kv_cache_memory_bytes": 134217728,
            }
        )
        tokenizer = SimpleNamespace(pad_token_id=0)
        model_config = SimpleNamespace(max_position_embeddings=32768)

        with (
            patch.object(vllm_rollout.torch.distributed, "get_world_size", return_value=1),
            patch.object(vllm_rollout, "LLM") as adapter_cls,
        ):
            vllm_rollout.vLLMRollout(None, config, tokenizer, model_config)

        self.assertEqual(adapter_cls.call_args.kwargs["max_num_batched_tokens"], 1792)
        self.assertEqual(adapter_cls.call_args.kwargs["max_num_seqs"], 2)
        self.assertEqual(
            adapter_cls.call_args.kwargs["kv_cache_memory_bytes"], 134217728
        )

    def test_supported_vllm_exports_legacy_rollout_surface(self):
        from verl.third_party.vllm import LLM, vllm_version

        self.assertEqual(vllm_version, "0.27.1")
        for method_name in (
            "generate",
            "sync_model_weights",
            "offload_model_weights",
            "init_cache_engine",
            "free_cache_engine",
        ):
            self.assertTrue(
                hasattr(LLM, method_name),
                f"compatibility LLM is missing {method_name}()",
            )

    def test_generate_preserves_legacy_tensor_output_contract(self):
        from verl.third_party.vllm import LLM

        class FakeEngine:
            def generate(self, prompts, sampling_params, use_tqdm, **kwargs):
                self.prompts = prompts
                return [
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                token_ids=[11, 12],
                                logprobs=[
                                    {11: SimpleNamespace(logprob=-0.1)},
                                    {12: SimpleNamespace(logprob=-0.2)},
                                ],
                            )
                        ]
                    ),
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                token_ids=[21],
                                logprobs=[{21: SimpleNamespace(logprob=-0.3)}],
                            )
                        ]
                    ),
                ]

        adapter = LLM.__new__(LLM)
        adapter.pad_token_id = 0
        adapter._engine = FakeEngine()

        token_ids, logprobs = adapter.generate(
            prompt_token_ids=[[1], [2]], sampling_params=object()
        )

        self.assertTrue(torch.equal(token_ids, torch.tensor([[11, 12], [21, 0]])))
        self.assertEqual(logprobs.shape, torch.Size([2, 2]))
        self.assertAlmostEqual(logprobs[0, 0].item(), -0.1, places=5)
        self.assertAlmostEqual(logprobs[1, 1].item(), 0.0, places=5)

    def test_weight_sync_wakes_weights_before_kv_cache(self):
        from verl.third_party.vllm import LLM

        events = []

        class FakeEngine:
            def wake_up(self, tags=None):
                events.append(("wake_up", tags))

            def collective_rpc(self, method, args):
                events.append(("collective_rpc", method))

            def reset_prefix_cache(self):
                events.append(("reset_prefix_cache", None))

        adapter = LLM.__new__(LLM)
        adapter._engine = FakeEngine()
        adapter._sleeping = True
        adapter.sync_model_weights({"weight": torch.zeros(1)})

        self.assertEqual(
            events,
            [
                ("wake_up", ["weights"]),
                ("collective_rpc", ANY),
                ("reset_prefix_cache", None),
                ("wake_up", ["kv_cache"]),
            ],
        )

    def test_offload_discards_stale_vllm_weights(self):
        from verl.third_party.vllm import LLM

        class FakeEngine:
            def __init__(self):
                self.sleep_call = None

            def sleep(self, **kwargs):
                self.sleep_call = kwargs

        adapter = LLM.__new__(LLM)
        adapter._engine = FakeEngine()
        adapter._sleeping = False
        adapter.offload_model_weights()

        self.assertEqual(adapter._engine.sleep_call, {"level": 2, "mode": "abort"})


if __name__ == "__main__":
    unittest.main()
