import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from medical_graphrag.retrieval.document_embeddings import (
    _coverage,
    _window_ranges,
    build_document_embeddings,
    embed_documents_full,
    ensure_document_embeddings,
)


class _MockTokenizer:
    def encode(self, text, add_special_tokens=True):
        toks = list(text)
        if add_special_tokens:
            toks = ["[CLS]"] + toks + ["[SEP]"]
        return toks

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(
            t for t in token_ids if not (skip_special_tokens and t.startswith("["))
        )


class _MockEmbedder:
    def __init__(self):
        self.tokenizer = _MockTokenizer()

    def get_max_seq_length(self):
        return 8  # 含 special tokens 上限 8

    def encode(self, texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            v = np.zeros(6, dtype=np.float32)
            for ch in t:
                v[ord(ch) % 6] += 1.0
            if normalize_embeddings:
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    def encode_token_windows(self, windows, *, batch_size, normalize_embeddings):
        vecs = []
        for window in windows:
            vector = np.zeros(6, dtype=np.float32)
            for token in window:
                vector[ord(token) % 6] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            vecs.append(vector)
        return np.asarray(vecs, dtype=np.float32)


class _NoRoundTripTokenizer(_MockTokenizer):
    def decode(self, token_ids, skip_special_tokens=True):
        raise AssertionError("token windows must not be decoded and re-tokenized")


class _TokenWindowEmbedder(_MockEmbedder):
    def __init__(self):
        self.tokenizer = _NoRoundTripTokenizer()
        self.encoded_windows = None

    def encode_token_windows(self, windows, *, batch_size, normalize_embeddings):
        self.encoded_windows = [list(window) for window in windows]
        vecs = []
        for window in windows:
            vector = np.zeros(6, dtype=np.float32)
            for token in window:
                vector[ord(token) % 6] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            vecs.append(vector)
        return np.asarray(vecs, dtype=np.float32)


def test_window_ranges_cover_all_positions():
    ranges = _window_ranges(9, max_window_len=4, overlap_tokens=1)
    full, truncated = _coverage(9, ranges)
    assert full is True
    assert truncated == 0
    assert ranges == [(0, 4), (3, 7), (6, 9)]  # 尾窗口不足但覆盖到 n


def test_window_ranges_duplicates_kept_by_position():
    # 重复 token 用位置覆盖证明,而不是 token 值集合
    n = 7
    ranges = _window_ranges(n, max_window_len=4, overlap_tokens=1)
    full, truncated = _coverage(n, ranges)
    assert full and truncated == 0


def test_window_ranges_single_window():
    ranges = _window_ranges(3, max_window_len=4, overlap_tokens=1)
    assert ranges == [(0, 3)]
    assert _coverage(3, ranges) == (True, 0)


def test_window_ranges_empty_text():
    assert _window_ranges(0, max_window_len=4, overlap_tokens=1) == []
    assert _coverage(0, []) == (True, 0)


def test_window_ranges_reject_bad_params():
    with pytest.raises(ValueError):
        _window_ranges(5, max_window_len=0, overlap_tokens=0)
    with pytest.raises(ValueError):
        _window_ranges(5, max_window_len=3, overlap_tokens=3)


def test_embed_documents_full_mean_then_l2():
    text = "ab" + "c" * 50  # 超过 mock max_seq_length → 多窗口
    embeddings, window_counts, truncated = embed_documents_full(
        [text], _MockEmbedder(), overlap_tokens=2
    )
    assert truncated == 0
    assert embeddings.shape[1] == 6
    assert window_counts[0] >= 2
    assert np.isclose(np.linalg.norm(embeddings[0]), 1.0, atol=1e-5)


def test_embed_documents_full_single_window_matches_direct_encode():
    embedder = _MockEmbedder()
    embeddings, counts, truncated = embed_documents_full(["ab"], embedder, overlap_tokens=0)
    assert truncated == 0 and counts == [1]
    expected = embedder.encode("ab", normalize_embeddings=True)[0]
    assert np.allclose(embeddings[0], expected, atol=1e-5)


def test_embed_documents_full_encodes_exact_token_id_windows_without_round_trip():
    embedder = _TokenWindowEmbedder()
    embeddings, counts, truncated = embed_documents_full(
        ["abcdefghij"], embedder, overlap_tokens=2
    )
    assert embeddings.shape == (1, 6)
    assert counts == [2]
    assert truncated == 0
    assert embedder.encoded_windows == [list("abcdef"), list("efghij")]


def test_embed_documents_full_rejects_empty_text():
    with pytest.raises(ValueError, match="empty"):
        embed_documents_full([""], _MockEmbedder(), overlap_tokens=1)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "ds"
    dataset.mkdir()
    (dataset / "documents.jsonl").write_text(
        "\n".join([
            json.dumps({"doc_id": "d1", "title": "t", "content": "abcabc",
                        "source": "s", "year": "y"}),
            json.dumps({"doc_id": "d2", "title": "t", "content": "xyzxyzxyz",
                        "source": "s", "year": "y"}),
        ]) + "\n",
        encoding="utf-8",
    )
    (dataset / "chunks.jsonl").write_text("", encoding="utf-8")
    (dataset / "questions.jsonl").write_text("", encoding="utf-8")
    (dataset / "qrels.tsv").write_text("query_id\tdoc_id\trelevance\n", encoding="utf-8")
    names = ("questions.jsonl", "documents.jsonl", "chunks.jsonl", "qrels.tsv")
    (dataset / "manifest.json").write_text(
        json.dumps({
            "counts": {"questions": 0, "documents": 2, "chunks": 0, "qrels": 0},
            "artifact_hashes": {name: _sha(dataset / name) for name in names},
        }),
        encoding="utf-8",
    )
    return dataset


def test_build_document_embeddings_artifact(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    out = tmp_path / "artifact"
    report = build_document_embeddings(
        dataset, out, model_name="mock", embedder=_MockEmbedder(), overlap_tokens=2
    )
    assert report["retrieval_unit"] == "document"
    assert report["token_window_encoding"] == "token_ids_direct_v1"
    assert report["document_count"] == 2
    assert report["window_coverage"]["truncated_token_count"] == 0
    embeddings = np.load(out / "document_embeddings.npy")
    assert embeddings.shape[0] == 2
    metadata = [json.loads(l) for l in
                (out / "document_embedding_metadata.jsonl").open(encoding="utf-8")]
    assert [m["doc_id"] for m in metadata] == ["d1", "d2"]
    assert report["embeddings_sha256"] == _sha(out / "document_embeddings.npy")
    assert report["metadata_sha256"] == _sha(out / "document_embedding_metadata.jsonl")


def test_ensure_document_embeddings_rejects_overlap_mismatch(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path)
    artifact = tmp_path / "artifact"
    build_document_embeddings(
        dataset,
        artifact,
        model_name="mock",
        embedder=_MockEmbedder(),
        overlap_tokens=2,
    )
    with pytest.raises(ValueError, match="overlap"):
        ensure_document_embeddings(
            dataset,
            artifact,
            model_name="mock",
            overlap_tokens=1,
        )
