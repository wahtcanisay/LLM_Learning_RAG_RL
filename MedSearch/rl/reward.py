import re
import string
import os

import aiohttp


def _normalize_answer(text: str) -> str:
    """统一可比较形式"""
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_acc_score(solution_str, ground_truth,):
    """从闭合 answer 中取最后一个答案，做宽松字符串命中"""
    matches = [answer for answer in re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL) if answer.strip() not in {"answer", "and"}]
    if not matches:
        return 0
    targets = ground_truth["target"]
    if isinstance(targets, str):
        targets = [targets]
    prediction = _normalize_answer(matches[-1].strip())
    for target in targets:
        target = _normalize_answer(target)
        if target and (" " + target + " ") in (" " + prediction + " "):
            return 1
    return 0

def compute_outcome_reward(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """参考DAPO, 通过计算acc{0,1}映射到{-1, +1}"""
    acc_score = compute_acc_score(solution_str, ground_truth)
    return {
        "score": 2 * acc_score - 1, 
        "acc" : acc_score
    }

async def compute_verifier_outcome_reward(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    **kwargs,
):
    """同步调用 CPU verifier，并把二分类结果作为 DAPO 的 acc 与 reward。"""
    answers = [
        answer.strip()
        for answer in re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
        if answer.strip() not in {"answer", "and"}
    ]
    if not answers:
        return {"score": -1, "acc": 0, "verifier_prob": 0.0}

    targets = ground_truth["target"]
    if isinstance(targets, str):
        targets = [targets]
    targets = [str(target) for target in targets if str(target).strip()]
    if not targets:
        return {"score": -1, "acc": 0, "verifier_prob": 0.0}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            os.environ.get("MEDSEARCH_VERIFIER_URL", "http://127.0.0.1:8002/verify"),
            json={"answer": answers[-1], "references": targets},
            timeout=aiohttp.ClientTimeout(total=300),
        ) as response:
            response.raise_for_status()
            result = await response.json()

    verifier_prob = float(result["prob"])
    acc_score = int(verifier_prob > 0.5)
    return {"score": 2 * acc_score - 1, "acc": acc_score, "verifier_prob": verifier_prob}
