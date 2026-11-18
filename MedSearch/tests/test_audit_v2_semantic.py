import torch

from MedSearch.tools.audit_v2_semantic import find_semantic_matches


def test_finds_history_and_internal_neighbors_without_self_matches():
    candidates = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    history = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    candidate_ids = ["a", "b", "c"]
    history_ids = ["old-a", "old-b"]

    matches = find_semantic_matches(candidates, history, candidate_ids, history_ids, 0.92)

    assert {tuple((item["kind"], item["candidate"], item["match"])) for item in matches} == {
        ("history", "a", "old-a"),
        ("internal", "b", "c"),
    }
