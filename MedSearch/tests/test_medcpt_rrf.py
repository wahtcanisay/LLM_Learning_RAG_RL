from MedSearch.tools.run_medcpt_retrieval_server import rrf_rank


def test_rrf_rewards_agreement_between_retrievers():
    ids, scores = rrf_rank(["1", "2", "3"], ["4", "2", "5"], topk=3)
    assert ids == ["2", "1", "4"]
    assert scores[0] > scores[1]


def test_rrf_can_return_dense_only_results():
    ids, _ = rrf_rank([], ["4", "2"], topk=2)
    assert ids == ["4", "2"]
