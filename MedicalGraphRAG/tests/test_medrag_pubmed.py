from pathlib import Path

from medical_graphrag.data.medrag_pubmed import content_hash, normalize_text, sample_distractors


SHARDS = Path(__file__).parent / "fixtures" / "medrag_pubmed"


def test_sample_distractors_is_repeatable() -> None:
    first, first_shards = sample_distractors(
        SHARDS,
        seed=7,
        shard_count=2,
        per_shard=3,
        target_count=4,
        excluded_titles=set(),
        excluded_content_hashes=set(),
    )
    second, second_shards = sample_distractors(
        SHARDS,
        seed=7,
        shard_count=2,
        per_shard=3,
        target_count=4,
        excluded_titles=set(),
        excluded_content_hashes=set(),
    )
    assert [item.doc_id for item in first] == [item.doc_id for item in second]
    assert first_shards == second_shards
    assert len(first) == 4
    assert all(item.source == "medrag_pubmed" for item in first)


def test_sample_distractors_applies_normalized_exclusions() -> None:
    excluded_titles = {normalize_text("  ALPHA  ")}
    excluded_hashes = {content_hash("Delta", "Delta abstract.")}
    documents, _ = sample_distractors(
        SHARDS,
        seed=7,
        shard_count=2,
        per_shard=3,
        target_count=4,
        excluded_titles=excluded_titles,
        excluded_content_hashes=excluded_hashes,
    )
    assert {item.title for item in documents}.isdisjoint({"Alpha", "Delta"})
