import re
import string


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Return Search-R1 exact-match reward using current veRL's response-only API."""
    matches = re.findall(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
    if not matches:
        return 0
    targets = ground_truth["target"]
    if isinstance(targets, str):
        targets = [targets]
    prediction = _normalize_answer(matches[-1].strip())
    return int(any(prediction == _normalize_answer(target) for target in targets))
