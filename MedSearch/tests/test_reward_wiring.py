import asyncio
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
VERL_ROOT = ROOT / "MedSearch" / ".deps" / "verl"
REWARD_PATH = ROOT / "MedSearch" / "rl" / "reward.py"
sys.path.insert(0, str(VERL_ROOT))

from verl import DataProto
from verl.trainer.ppo.reward import get_custom_reward_fn, load_reward_manager


class _Tokenizer:
    def __init__(self, response: str):
        self.response = response

    def decode(self, token_ids, skip_special_tokens=True):
        return self.response


def _config():
    return OmegaConf.create(
        {
            "reward": {
                "custom_reward_function": {
                    "path": str(REWARD_PATH),
                    "name": "compute_outcome_reward",
                    "reward_kwargs": {},
                },
                "reward_manager": {"source": "register", "name": "dapo"},
                "reward_kwargs": {"overlong_buffer_cfg": {"enable": False}},
            }
        }
    )


def _data():
    return DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        },
        non_tensors={
            "data_source": np.array(["medsearch"], dtype=object),
            "reward_model": np.array([{"ground_truth": {"target": ["Beijing"]}}], dtype=object),
        },
    )


def test_verl_loads_medsearch_reward_from_config():
    reward_fn = get_custom_reward_fn(_config())

    assert reward_fn is not None
    assert reward_fn("medsearch", "<answer>Beijing</answer>", {"target": ["Beijing"]}) == {
        "score": 1,
        "acc": 1,
    }


def test_reward_uses_one_way_whole_word_substring_match():
    reward_fn = get_custom_reward_fn(_config())
    assert reward_fn("medsearch", "<answer>trastuzumab plus chemotherapy</answer>", {"target": ["trastuzumab"]})["acc"] == 1
    assert reward_fn("medsearch", "<answer>right lower lobe</answer>", {"target": ["Right lower lobe apical"]})["acc"] == 0
    assert reward_fn("medsearch", "<answer>polyunsaturated fatty acids</answer>", {"target": ["saturated fatty acids"]})["acc"] == 0


def test_dapo_reward_manager_preserves_score_and_acc():
    async def run_reward_manager():
        manager = load_reward_manager(_config(), _Tokenizer("<answer>Beijing</answer>"))
        return await manager.run_single(_data())

    result = asyncio.run(run_reward_manager())

    assert result == {
        "reward_score": 1,
        "reward_extra_info": {"score": 1, "acc": 1},
    }
