import unittest

from verl.workers.fsdp_workers import _get_attention_backend


class AttentionBackendTest(unittest.TestCase):
    def test_uses_flash_attention_when_available(self):
        self.assertEqual(_get_attention_backend(True), "flash_attention_2")

    def test_falls_back_to_sdpa_without_flash_attention(self):
        self.assertEqual(_get_attention_backend(False), "sdpa")


if __name__ == "__main__":
    unittest.main()
