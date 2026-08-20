import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "experiments" / "single_gpu_smoke" / "run_grpo_1step_modern_verl.sh"
SETUP_SCRIPT = ROOT / "experiments" / "single_gpu_smoke" / "setup_modern_verl.sh"
RUNTIME_REQUIREMENTS = ROOT / "modern_verl" / "requirements-runtime.txt"
VERL_SHM_PATCHER = ROOT / "modern_verl" / "patch_verl_for_wsl.py"


class ModernSmokeConfigTest(unittest.TestCase):
    def test_setup_pins_upstream_verl_without_reinstalling_torch_or_vllm(self):
        text = SETUP_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("e78efedbe56591cd53ac107150abb7565bb9b8b1", text)
        self.assertIn('-c safe.directory="${VERL_DIR}"', text)
        self.assertIn("pwd -P", text)
        self.assertIn("--no-deps", text)
        self.assertEqual(text.count("--break-system-packages"), 3)
        self.assertNotIn("pip install torch", text)
        self.assertNotIn("pip install vllm", text)
        requirements = RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")
        self.assertIn("transformers>=5.5.3,<5.11,!=5.6.0", requirements)
        self.assertIn("tensordict==0.10.0", requirements)
        self.assertIn("torchdata", requirements)
        self.assertIn("peft", requirements)

    def test_setup_applies_wsl_safe_shared_memory_weight_transfer_patch(self):
        setup_text = SETUP_SCRIPT.read_text(encoding="utf-8")
        patcher_text = VERL_SHM_PATCHER.read_text(encoding="utf-8")
        self.assertIn("patch_verl_for_wsl.py", setup_text)
        self.assertIn("VERL_FORCE_SHM_WEIGHT_TRANSFER", patcher_text)
        self.assertIn('!= \"1\"', patcher_text)

    def test_launcher_keeps_the_required_single_gpu_3b_grpo_gate(self):
        text = RUN_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("pwd -P", text)
        self.assertIn("export VERL_FORCE_SHM_WEIGHT_TRANSFER=1", text)
        required = [
            "verl.trainer.main_ppo",
            "trainer.use_v1=False",
            "Qwen2.5-3B-Instruct",
            "algorithm.adv_estimator=grpo",
            "data.dataloader_num_workers=0",
            "actor_rollout_ref.model.lora_rank=16",
            "actor_rollout_ref.model.target_modules=all-linear",
            "actor_rollout_ref.model.use_remove_padding=False",
            "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
            "actor_rollout_ref.actor.strategy=fsdp2",
            "actor_rollout_ref.actor.ppo_mini_batch_size=1",
            "actor_rollout_ref.rollout.load_format=safetensors",
            "actor_rollout_ref.rollout.layered_summon=True",
            "actor_rollout_ref.rollout.n=2",
            "actor_rollout_ref.rollout.agent.num_workers=2",
            "actor_rollout_ref.actor.use_kl_loss=True",
            "trainer.n_gpus_per_node=1",
            "trainer.total_training_steps=1",
            "search_r1_agent",
            "search_r1_modern/reward.py",
        ]
        for value in required:
            self.assertIn(value, text)
        self.assertNotIn("1.5B", text)


if __name__ == "__main__":
    unittest.main()
