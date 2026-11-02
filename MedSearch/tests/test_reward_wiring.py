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
from verl.trainer.ppo.reward import get_custom_reward_fn, load_reward_manager, resolve_reward_manager_cls


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


def test_grpo_launcher_manager_forwards_final_answer_to_callback():
    import re

    launcher = (ROOT / "MedSearch/scripts/run_medsearch_grpo_distill.sh").read_text(encoding="utf-8")
    config = _config()
    config.reward.reward_manager.name = re.search(r"reward\.reward_manager\.name=(\w+)", launcher).group(1)

    async def run():
        seen = []

        async def score(data_source, solution_str, ground_truth, extra_info):
            assert extra_info["source_note"] == "preserved"
            answer = extra_info["final_answer"]
            seen.append(answer)
            acc = int(answer == "Beijing")
            return {"score": 2 * acc - 1, "acc": acc}

        manager = resolve_reward_manager_cls(config)(
            config, _Tokenizer("<information><answer>Shanghai</answer></information>"), score,
        )
        for answer, expected in [("Beijing", 1), (None, -1)]:
            data = _data()
            data.non_tensor_batch["extra_info"] = np.array([{"source_note": "preserved"}], dtype=object)
            data.non_tensor_batch["tool_extra_fields"] = np.array([{"final_answer": answer}], dtype=object)
            result = await manager.run_single(data)
            assert result["reward_score"] == expected
            assert result["reward_extra_info"]["acc"] == int(expected == 1)
        assert seen == ["Beijing", None]

    asyncio.run(run())


def test_verifier_reward_uses_only_final_answer_over_http(monkeypatch):
    from aiohttp import web
    from aiohttp.test_utils import TestServer

    async def run():
        requests = []

        async def verify(request):
            requests.append(await request.json())
            return web.json_response({"prob": 0.9})

        app = web.Application()
        app.router.add_post("/verify", verify)
        async with TestServer(app) as server:
            monkeypatch.setenv("MEDSEARCH_VERIFIER_URL", str(server.make_url("/verify")))
            config = _config()
            config.reward.reward_manager.name = "naive"
            config.reward.custom_reward_function.name = "compute_verifier_outcome_reward"
            manager = load_reward_manager(
                config,
                _Tokenizer("<information><answer>Beijing</answer></information><answer>Shanghai</answer>"),
            )
            for answer in [None, "", "Beijing"]:
                data = _data()
                data.non_tensor_batch["tool_extra_fields"] = np.array([{"final_answer": answer}], dtype=object)
                result = await manager.run_single(data)
                if not answer:
                    assert requests == []
                    assert result == {
                        "reward_score": -1,
                        "reward_extra_info": {"score": -1, "acc": 0, "verifier_prob": 0.0},
                    }
                else:
                    assert requests == [{"answer": "Beijing", "references": ["Beijing"]}]
                    assert result == {
                        "reward_score": 1,
                        "reward_extra_info": {"score": 1, "acc": 1, "verifier_prob": 0.9},
                    }

    asyncio.run(run())
