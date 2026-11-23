import unittest

from omegaconf import OmegaConf

from verl.trainer.main_ppo import _should_create_reference_policy


class ValidationWorkerRolesTest(unittest.TestCase):
    def test_validation_only_does_not_create_reference_policy(self):
        config = OmegaConf.create({'trainer': {'val_only': True}})
        self.assertFalse(_should_create_reference_policy(config))

    def test_training_keeps_reference_policy(self):
        config = OmegaConf.create({'trainer': {'val_only': False}})
        self.assertTrue(_should_create_reference_policy(config))


if __name__ == '__main__':
    unittest.main()
