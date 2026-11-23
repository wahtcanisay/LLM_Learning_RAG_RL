"""Regression test for the actor's optional FlashAttention padding helpers."""

import importlib
import unittest


class OptionalFlashAttentionImportTest(unittest.TestCase):
    def test_actor_module_imports_without_flash_attn(self):
        module = importlib.import_module("verl.workers.actor.dp_actor")
        self.assertTrue(hasattr(module, "DataParallelPPOActor"))


if __name__ == "__main__":
    unittest.main()
