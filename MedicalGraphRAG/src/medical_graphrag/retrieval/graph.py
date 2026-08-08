"""LinearGraphRetriever: relation-free graph retrieval, a faithful port of
LinearRAG's default non-vectorized retrieval core, adapted to the frozen chunk
data and the shared evaluation contract.

Offline build (one pass): medical NER over passage content -> entities;
sentence split + sentence embeddings -> Entity<->Sentence bridge;
Entity-Passage edges (normalized co-occurrence) -> igraph with Entity and
Passage nodes -> persisted graph + stores + hashed report.

Two retrieval units (spec §3/§5):
  * ``document``: one passage per frozen ``documents.jsonl`` row, passage_id =
    doc_id, full abstract content. Document embeddings come from the SHARED
    frozen artifact (``document_embeddings.py``) consumed by Dense, Similarity
    kNN and Graph passage prior alike (P0-8).
  * ``chunk``: one passage per frozen ``chunks.jsonl`` row (historical default).

Passage-Passage edges are governed by ``GraphBuildConfig.passage_edge_mode``,
mutually exclusive: ``none`` | ``similarity`` | ``adjacent`` (spec §6.3).

Online search (per query): question NER -> seed entities (argmax over corpus
entity embeddings) -> Entity->Sentence->Entity BFS propagation (semantic
bridge) -> passage prior (normalized Dense + activated-entity reward) ->
Personalized PageRank -> top-k passage (doc/chunk) ids.
"""
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from medical_graphrag.data.io import sha256_file, write_json, write_jsonl
from medical_graphrag.data.retrieval_passages import load_retrieval_passages
from medical_graphrag.retrieval.bm25 import validate_frozen_dataset

DEFAULT_EMBEDDING_MODEL = "models/all-mpnet-base-v2"
DEFAULT_NER_MODEL = "en_ner_bc5cdr_md"
TEXT_MODE = "abstract_only"

GRAPH_SCHEMA_VERSION = 2


class GraphConfig:
    """Online retrieval (PPR) hyper-parameters, kept separate from the offline
    ``GraphBuildConfig`` (spec §9). Aligned to LinearRAG official defaults."""

    def __init__(
        self,
        *,
        damping: float = 0.5,
        passage_ratio: float = 2.0,
        passage_node_weight: float = 0.05,
        iteration_threshold: float = 0.5,
        top_k_sentence: int = 1,
        max_iterations: int = 3,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        ner_model: str = DEFAULT_NER_MODEL,
    ) -> None:
        self.damping = damping
        self.passage_ratio = passage_ratio
        self.passage_node_weight = passage_node_weight
        self.iteration_threshold = iteration_threshold
        self.top_k_sentence = top_k_sentence
        self.max_iterations = max_iterations
        self.embedding_model = embedding_model
        self.ner_model = ner_model


@dataclass(frozen=True)
class GraphBuildConfig:
    """Offline graph-construction parameters (edge strategy), separate from the
    online PPR ``GraphConfig`` (spec §9).

    Validation (spec §9): document unit only supports ``none``/``similarity``;
    chunk unit only supports ``none``/``adjacent``; similarity params must be in
    range; ``window_overlap_tokens`` must be non-negative.
    """

    retrieval_unit: Literal["document", "chunk"]
    passage_edge_mode: Literal["none", "similarity", "adjacent"]
    embedding_model: str
    ner_model: str
    similarity_k: int = 5
    similarity_min_cosine: float = 0.50
    similarity_edge_scale: float = 1.0
    window_overlap_tokens: int = 32

    def __post_init__(self) -> None:
        if self.retrieval_unit == "document":
            if self.passage_edge_mode not in {"none", "similarity"}:
                raise ValueError(
                    "document retrieval_unit only supports none/similarity edge modes"
                )
        elif self.retrieval_unit == "chunk":
            if self.passage_edge_mode not in {"none", "adjacent"}:
                raise ValueError(
                    "chunk retrieval_unit only supports none/adjacent edge modes"
                )
        else:
            raise ValueError("retrieval_unit must be 'document' or 'chunk'")
        if self.similarity_k < 1:
            raise ValueError("similarity_k must be >= 1")
        if not (0 <= self.similarity_min_cosine <= 1):
            raise ValueError("similarity_min_cosine must be in [0, 1]")
        if self.similarity_edge_scale <= 0:
            raise ValueError("similarity_edge_scale must be > 0")
        if self.window_overlap_tokens < 0:
            raise ValueError("window_overlap_tokens must be >= 0")

    @property
    def graph_profile(self) -> str:
        if self.retrieval_unit == "document":
            return (
                "document_ep_v1"
                if self.passage_edge_mode == "none"
                else "document_similarity_v1"
            )
        if self.passage_edge_mode == "adjacent":
            return "linearrag_adjacent_v1"
        return "chunk_entity_only_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _load_nlp(model_name: str):
    import spacy

    nlp = spacy.load(model_name)
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def extract_entities(nlp, texts: list[str], *, batch_size: int = 64) -> list[list[str]]:
    """Return a per-text list of de-duplicated, order-preserving entity strings.

    Ordinal/cardinal entities are excluded, matching LinearRAG's official
    ``extract_entities_sentences`` (they are numerous and low-discrimination).
    """
    results: list[list[str]] = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        seen: set[str] = set()
        entities: list[str] = []
        for ent in doc.ents:
            if ent.label_ in ("ORDINAL", "CARDINAL"):
                continue
            text = ent.text.strip()
            if text and text not in seen:
                seen.add(text)
                entities.append(text)
        results.append(entities)
    return results


def split_sentences(nlp, texts: list[str]) -> list[list[str]]:
    """Split each text into non-empty sentences using the pipeline's sents."""
    results: list[list[str]] = []
    for doc in nlp.pipe(texts, batch_size=64):
        results.append([s.text.strip() for s in doc.sents if s.text.strip()])
    return results


def build_entity_passage_edges(
    passage_ids: list[str],
    passage_texts: list[str],
    passage_entities: list[list[str]],
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Compute Entity-Passage edge weights (normalized occurrence counts).

    Matches LinearRAG's ``add_entity_to_passage_edges``: an entity's weight is
    its string-occurrence count in the passage text divided by the sum of
    occurrence counts of all entities in that passage.
    """
    edges: dict[str, dict[str, float]] = {}
    all_entities: set[str] = set()
    for passage_id, text, entities in zip(passage_ids, passage_texts, passage_entities):
        if not entities:
            continue
        counts = {entity: text.count(entity) for entity in entities}
        total = sum(counts.values())
        if total == 0:
            continue
        edges[passage_id] = {e: c / total for e, c in counts.items()}
        all_entities.update(counts)
    return sorted(all_entities), edges


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    k = int((len(sorted_values) - 1) * q)
    return float(sorted_values[k])


def _hash_dict(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _weight_stats(weights: list[float]) -> dict[str, float]:
    if not weights:
        return {"count": 0}
    return {
        "count": float(len(weights)),
        "sum": float(sum(weights)),
        "min": float(min(weights)),
        "mean": float(sum(weights) / len(weights)),
        "p50": _percentile(sorted(weights), 0.50),
        "p95": _percentile(sorted(weights), 0.95),
        "max": float(max(weights)),
    }


def _count_exact_content_duplicates(passages) -> int:
    from collections import Counter

    counts = Counter(p.content for p in passages)
    return sum(c * (c - 1) // 2 for c in counts.values() if c > 1)


def _load_document_embedding_artifact(
    document_embeddings_dir: Path,
    dataset_dir: Path,
    passage_count: int,
    *,
    expected_model_name: str,
    expected_overlap_tokens: int,
) -> tuple[np.ndarray, dict[str, Any], str, str]:
    """Load the frozen document embedding artifact and validate its bindings.

    Returns ``(embeddings, artifact_report, embedding_report_sha256,
    embeddings_sha256)``. Fails closed on any dataset/manifest/count/hash
    mismatch (P0-8).
    """
    from medical_graphrag.retrieval.document_embeddings import (
        load_document_embedding_artifact,
    )

    embeddings, _doc_ids, report, report_sha256 = load_document_embedding_artifact(
        dataset_dir,
        document_embeddings_dir,
        expected_model_name=expected_model_name,
        expected_overlap_tokens=expected_overlap_tokens,
    )
    if embeddings.shape[0] != passage_count:
        raise ValueError("embedding row count does not match passages")
    return (
        embeddings,
        report,
        report_sha256,
        report["embeddings_sha256"],
    )


def build_graph_index(
    dataset_dir: Path,
    output_dir: Path,
    *,
    build_config: GraphBuildConfig | None = None,
    config: GraphConfig | None = None,
    batch_size: int = 64,
    embedder: Any | None = None,
    nlp: Any | None = None,
    document_embeddings_dir: Path | None = None,
) -> dict[str, Any]:
    """Run NER + sentence bridge + edges, build and persist the igraph.

    ``build_config`` selects the retrieval unit and the Passage-Passage edge
    mode. ``config`` is the online PPR ``GraphConfig`` (kept for backward
    compatibility with historical runners). For the document unit,
    ``document_embeddings_dir`` must point at the frozen artifact from
    ``build_document_embeddings`` (P0-8).
    """
    import faiss
    import igraph as ig

    if build_config is None:
        build_config = GraphBuildConfig(
            retrieval_unit="chunk",
            passage_edge_mode="none",
            embedding_model=(config.embedding_model if config else DEFAULT_EMBEDDING_MODEL),
            ner_model=(config.ner_model if config else DEFAULT_NER_MODEL),
        )
    if config is None:
        config = GraphConfig(
            ner_model=build_config.ner_model,
            embedding_model=build_config.embedding_model,
        )

    manifest = validate_frozen_dataset(dataset_dir)
    passages = load_retrieval_passages(dataset_dir, build_config.retrieval_unit)
    source_artifact = (
        "documents.jsonl"
        if build_config.retrieval_unit == "document"
        else "chunks.jsonl"
    )
    expected_count = manifest["counts"]["documents" if build_config.retrieval_unit == "document" else "chunks"]
    if len(passages) != expected_count:
        raise ValueError("passage count does not match frozen dataset count")
    passage_ids = [p.passage_id for p in passages]
    texts = [p.content for p in passages]
    doc_ids = [p.doc_id for p in passages]
    orders = [p.order for p in passages]

    nlp = nlp if nlp is not None else _load_nlp(build_config.ner_model)
    passage_entities = extract_entities(nlp, texts, batch_size=batch_size)
    sentences = split_sentences(nlp, texts)

    # --- embeddings -------------------------------------------------------
    embedding_report_sha256 = None
    embedding_embeddings_sha256 = None
    if build_config.retrieval_unit == "document":
        if document_embeddings_dir is None:
            raise ValueError(
                "document retrieval_unit requires document_embeddings_dir"
            )
        passage_embeddings, artifact_report, embedding_report_sha256, embedding_embeddings_sha256 = (
            _load_document_embedding_artifact(
                document_embeddings_dir,
                dataset_dir,
                len(passages),
                expected_model_name=build_config.embedding_model,
                expected_overlap_tokens=build_config.window_overlap_tokens,
            )
        )
        window_coverage = artifact_report.get("window_coverage", {})
    else:
        embedder = embedder if embedder is not None else _load_embedder(build_config.embedding_model)
        passage_embeddings = np.asarray(
            embedder.encode(
                texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            ),
            dtype="float32",
        )
        window_coverage = {}

    # --- sentence bridge ----------------------------------------------------
    sentence_ids: list[str] = []
    sentence_texts: list[str] = []
    sentence_to_entities: list[list[str]] = []
    entity_to_sentences: dict[str, list[str]] = defaultdict(list)
    for p_idx, sentence_list in enumerate(sentences):
        passage_ents = passage_entities[p_idx]
        for s_idx, sentence in enumerate(sentence_list):
            sid = f"{passage_ids[p_idx]}#s{s_idx}"
            sentence_ids.append(sid)
            sentence_texts.append(sentence)
            sentence_entities = [e for e in passage_ents if e in sentence]
            sentence_to_entities.append(sentence_entities)
            for entity in sentence_entities:
                entity_to_sentences[entity].append(sid)

    if embedder is None:
        embedder = _load_embedder(build_config.embedding_model)
    sentence_embeddings = np.asarray(
        embedder.encode(
            sentence_texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ),
        dtype="float32",
    )

    # --- entity-passage edges + entity embeddings --------------------------
    all_entities, edges = build_entity_passage_edges(passage_ids, texts, passage_entities)
    entity_embeddings = np.asarray(
        embedder.encode(
            all_entities,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ),
        dtype="float32",
    )

    # --- igraph: Entity nodes then Passage nodes ----------------------------
    graph = ig.Graph(directed=False)
    entity_index = {e: i for i, e in enumerate(all_entities)}
    passage_index = {p: len(all_entities) + i for i, p in enumerate(passage_ids)}
    graph.add_vertices(len(all_entities) + len(passage_ids))
    graph.vs["name"] = list(all_entities) + passage_ids
    graph.vs["content"] = list(all_entities) + texts
    edge_list: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    for passage_id, entity_scores in edges.items():
        for entity, weight in entity_scores.items():
            edge_list.append((passage_index[passage_id], entity_index[entity]))
            edge_weights.append(weight)
    entity_passage_edge_count = len(edge_list)

    # --- passage-passage edges (mutually exclusive modes) -------------------
    passage_passage_edge_count = 0
    edge_count_by_type = {"entity_passage": entity_passage_edge_count, "similarity": 0, "adjacent": 0}
    passage_passage_diagnostics: dict[str, float] = {}
    from medical_graphrag.retrieval.graph_edges import (
        build_adjacent_edges,
        build_similarity_edges,
    )

    if build_config.passage_edge_mode == "similarity":
        sim_edges = build_similarity_edges(
            passage_ids,
            passage_embeddings,
            k=build_config.similarity_k,
            min_cosine=build_config.similarity_min_cosine,
            scale=build_config.similarity_edge_scale,
        )
        index_of = {pid: i for i, pid in enumerate(passage_ids)}
        degree = [0] * len(passage_ids)
        for a, b, _w in sim_edges:
            edge_list.append((passage_index[a], passage_index[b]))
            edge_weights.append(_w)
            degree[index_of[a]] += 1
            degree[index_of[b]] += 1
        passage_passage_edge_count = len(sim_edges)
        edge_count_by_type["similarity"] = len(sim_edges)
        sim_weights = [w for _, _, w in sim_edges]
        isolated = [pid for pid, d in zip(passage_ids, degree) if d == 0]
        passage_passage_diagnostics = {
            "edge_count": float(len(sim_edges)),
            "isolated_count": float(len(isolated)),
            "isolated_rate": float(len(isolated)) / max(len(passage_ids), 1),
            "degree_min": _percentile(sorted(degree), 0.0),
            "degree_mean": sum(degree) / max(len(degree), 1),
            "degree_p50": _percentile(sorted(degree), 0.50),
            "degree_p95": _percentile(sorted(degree), 0.95),
            "degree_p99": _percentile(sorted(degree), 0.99),
            "degree_max": _percentile(sorted(degree), 1.0),
            "weight_min": _percentile(sorted(sim_weights), 0.0),
            "weight_mean": sum(sim_weights) / max(len(sim_weights), 1),
            "weight_p50": _percentile(sorted(sim_weights), 0.50),
            "weight_p95": _percentile(sorted(sim_weights), 0.95),
            "weight_max": _percentile(sorted(sim_weights), 1.0),
            "exact_content_duplicate_pairs": float(_count_exact_content_duplicates(passages)),
        }
    elif build_config.passage_edge_mode == "adjacent":
        adj_edges, gaps = build_adjacent_edges(passages)
        doc_counts: dict[str, int] = defaultdict(int)
        for p in passages:
            doc_counts[p.doc_id] += 1
        candidate_adjacent = sum(max(c - 1, 0) for c in doc_counts.values())
        expected_adjacent = candidate_adjacent - len(gaps)
        if expected_adjacent != len(adj_edges):
            raise ValueError(
                f"adjacent expected {expected_adjacent} != actual {len(adj_edges)}"
            )
        for a, b, w in adj_edges:
            edge_list.append((passage_index[a], passage_index[b]))
            edge_weights.append(w)
        passage_passage_edge_count = len(adj_edges)
        edge_count_by_type["adjacent"] = len(adj_edges)
        passage_passage_diagnostics = {
            "candidate_pair_count": float(candidate_adjacent),
            "expected_edge_count": float(expected_adjacent),
            "actual_edge_count": float(len(adj_edges)),
            "gap_count": float(len(gaps)),
        }

    if edge_list:
        graph.add_edges(edge_list)
        graph.es["weight"] = edge_weights
    passage_node_indices = [passage_index[p] for p in passage_ids]

    # --- persist -------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "graph.graphml"
    graph.write_graphml(str(graph_path))
    write_jsonl(output_dir / "entities.jsonl", [{"entity": e} for e in all_entities])
    write_jsonl(
        output_dir / "entity_to_sentences.jsonl",
        [{"entity": e, "sentences": entity_to_sentences[e]} for e in all_entities],
    )
    write_jsonl(
        output_dir / "sentence_to_entities.jsonl",
        [
            {"sentence_id": sid, "entities": ents}
            for sid, ents in zip(sentence_ids, sentence_to_entities)
        ],
    )
    np.save(output_dir / "sentence_embeddings.npy", sentence_embeddings)
    np.save(output_dir / "entity_embeddings.npy", entity_embeddings)
    np.save(output_dir / "passage_embeddings.npy", passage_embeddings)

    report: dict[str, Any] = {
        "graph_schema_version": GRAPH_SCHEMA_VERSION,
        "graph_profile": build_config.graph_profile,
        "retrieval_unit": build_config.retrieval_unit,
        "passage_edge_mode": build_config.passage_edge_mode,
        "source_artifact": source_artifact,
        "source_artifact_sha256": manifest["artifact_hashes"][source_artifact],
        "text_mode": TEXT_MODE,
        "ner_model": build_config.ner_model,
        "embedding_model": build_config.embedding_model,
        "entity_count": len(all_entities),
        "passage_count": len(passage_ids),
        "sentence_count": len(sentence_ids),
        "edge_count": len(edge_list),
        "entity_passage_edge_count": entity_passage_edge_count,
        "passage_passage_edge_count": passage_passage_edge_count,
        "edge_count_by_type": edge_count_by_type,
        "edge_weight_stats_by_type": {
            "entity_passage": _weight_stats(edge_weights[:entity_passage_edge_count]),
            "similarity": (
                _weight_stats(edge_weights[entity_passage_edge_count:])
                if build_config.passage_edge_mode == "similarity"
                else {"count": 0}
            ),
            "adjacent": (
                _weight_stats(edge_weights[entity_passage_edge_count:])
                if build_config.passage_edge_mode == "adjacent"
                else {"count": 0}
            ),
        },
        "passage_passage_diagnostics": passage_passage_diagnostics,
        "adjacent_passage_edges": build_config.passage_edge_mode == "adjacent",
        "window_coverage": window_coverage,
        "embedding_report_sha256": embedding_report_sha256,
        "embedding_embeddings_sha256": embedding_embeddings_sha256,
        "build_config": {
            **build_config.to_dict(),
            "config_sha256": _hash_dict(build_config.to_dict()),
        },
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "dataset_artifact_hashes": manifest["artifact_hashes"],
        "graph_sha256": sha256_file(graph_path),
        "sentence_embeddings_sha256": sha256_file(output_dir / "sentence_embeddings.npy"),
        "entity_embeddings_sha256": sha256_file(output_dir / "entity_embeddings.npy"),
        "passage_embeddings_sha256": sha256_file(output_dir / "passage_embeddings.npy"),
        "entities_sha256": sha256_file(output_dir / "entities.jsonl"),
        "entity_to_sentences_sha256": sha256_file(output_dir / "entity_to_sentences.jsonl"),
        "sentence_to_entities_sha256": sha256_file(output_dir / "sentence_to_entities.jsonl"),
        "config": {
            "damping": config.damping,
            "passage_ratio": config.passage_ratio,
            "passage_node_weight": config.passage_node_weight,
            "iteration_threshold": config.iteration_threshold,
            "top_k_sentence": config.top_k_sentence,
            "max_iterations": config.max_iterations,
        },
    }
    write_json(output_dir / "graph_build.json", report)
    return report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class LinearGraphRetriever:
    """Online graph retrieval over a built graph index.

    Ports LinearRAG's default non-vectorized retrieval: question NER -> seed
    entities (argmax entity matching), Entity->Sentence->Entity BFS propagation,
    passage prior (normalized Dense + activated-entity reward), then PPR. When
    the question yields no entities, falls back to Dense passage ranking.
    """

    def __init__(
        self,
        index_dir: Path,
        *,
        config: GraphConfig | None = None,
        embedder: Any | None = None,
        nlp: Any | None = None,
    ):
        import igraph as ig

        if config is None:
            config = GraphConfig()
        self.config = config
        self.index_dir = Path(index_dir)
        self.report = json.loads(
            (self.index_dir / "graph_build.json").read_text(encoding="utf-8")
        )
        self.retrieval_unit = str(self.report.get("retrieval_unit", "chunk"))
        self.passage_edge_mode = str(self.report.get("passage_edge_mode", "none"))
        self.embedder = embedder if embedder is not None else _load_embedder(config.embedding_model)
        self.nlp = nlp if nlp is not None else _load_nlp(config.ner_model)
        self.graph = ig.Graph.Read_GraphML(str(self.index_dir / "graph.graphml"))
        self.entity_list = [
            row["entity"] for row in _read_jsonl(self.index_dir / "entities.jsonl")
        ]
        self.entity_embeddings = np.load(self.index_dir / "entity_embeddings.npy")
        self.sentence_embeddings = np.load(self.index_dir / "sentence_embeddings.npy")
        self.passage_embeddings = np.load(self.index_dir / "passage_embeddings.npy")
        self.sentence_to_entities = [
            row["entities"]
            for row in _read_jsonl(self.index_dir / "sentence_to_entities.jsonl")
        ]
        self.entity_to_sentences = {
            row["entity"]: row["sentences"]
            for row in _read_jsonl(self.index_dir / "entity_to_sentences.jsonl")
        }
        entity_count = int(self.report["entity_count"])
        names = [str(n) for n in self.graph.vs["name"]]
        contents = [str(c) for c in self.graph.vs["content"]]
        self.passage_ids = names[entity_count:]
        self.passage_texts = contents[entity_count:]
        self.passage_node_indices = list(range(entity_count, len(names)))
        self.node_name_to_vertex_idx = {name: i for i, name in enumerate(names)}
        self.sentence_id_to_idx = {
            str(row["sentence_id"]): i
            for i, row in enumerate(
                _read_jsonl(self.index_dir / "sentence_to_entities.jsonl")
            )
        }

    def _question_entities(self, query: str) -> list[str]:
        doc = self.nlp(query)
        seen: set[str] = set()
        entities: list[str] = []
        for ent in doc.ents:
            if ent.label_ in ("ORDINAL", "CARDINAL"):
                continue
            text = ent.text.strip()
            if text and text not in seen:
                seen.add(text)
                entities.append(text)
        return entities

    def _get_seed_entities(self, query: str) -> tuple[list[int], list[float]]:
        """Match each question entity to its argmax corpus entity."""
        question_entities = self._question_entities(query)
        if not question_entities or self.entity_embeddings.shape[0] == 0:
            return [], []
        q_embs = np.asarray(
            self.embedder.encode(
                question_entities, normalize_embeddings=True, show_progress_bar=False
            ),
            dtype="float32",
        )
        similarities = np.dot(self.entity_embeddings, q_embs.T)
        indices: list[int] = []
        scores: list[float] = []
        for i in range(similarities.shape[1]):
            best = int(np.argmax(similarities[:, i]))
            indices.append(best)
            scores.append(float(similarities[best, i]))
        return indices, scores

    def _calculate_entity_scores(
        self,
        query_embedding: np.ndarray,
        seed_entity_indices: list[int],
        seed_entity_scores: list[float],
    ) -> tuple[np.ndarray, dict[str, tuple[int, float, int]]]:
        """BFS Entity->Sentence->Entity propagation (port of LinearRAG)."""
        n_nodes = len(self.graph.vs["name"])
        entity_weights = np.zeros(n_nodes)
        actived: dict[str, tuple[int, float, int]] = {}
        for entity_idx, score in zip(seed_entity_indices, seed_entity_scores):
            entity = self.entity_list[entity_idx]
            node_idx = self.node_name_to_vertex_idx[entity]
            actived[entity] = (node_idx, score, 1)
            entity_weights[node_idx] = score
        used_sentence_ids: set[str] = set()
        current = dict(actived)
        iteration = 1
        while current and iteration < self.config.max_iterations:
            new: dict[str, tuple[int, float, int]] = {}
            for entity, (node_idx, entity_score, tier) in current.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                sentence_ids = [
                    sid
                    for sid in self.entity_to_sentences.get(entity, [])
                    if sid not in used_sentence_ids
                ]
                if not sentence_ids:
                    continue
                s_indices = [self.sentence_id_to_idx[sid] for sid in sentence_ids]
                s_embs = self.sentence_embeddings[s_indices]
                q_emb = (
                    query_embedding.reshape(-1, 1)
                    if query_embedding.ndim == 1
                    else query_embedding
                )
                similarities = np.dot(s_embs, q_emb).flatten()
                top = np.argsort(similarities)[::-1][: self.config.top_k_sentence]
                for pos in top:
                    sid = sentence_ids[int(pos)]
                    used_sentence_ids.add(sid)
                    for next_entity in self.sentence_to_entities[
                        self.sentence_id_to_idx[sid]
                    ]:
                        next_score = entity_score * float(similarities[int(pos)])
                        if next_score < self.config.iteration_threshold:
                            continue
                        next_node = self.node_name_to_vertex_idx[next_entity]
                        entity_weights[next_node] += next_score
                        new[next_entity] = (next_node, next_score, iteration + 1)
            actived.update(new)
            current = new
            iteration += 1
        return entity_weights, actived

    def _calculate_passage_scores(
        self, query_embedding: np.ndarray, actived_entities: dict[str, tuple[int, float, int]]
    ) -> np.ndarray:
        """Passage prior = ratio * normalized Dense + log(1 + entity reward)."""
        n_nodes = len(self.graph.vs["name"])
        passage_weights = np.zeros(n_nodes)
        query_emb = query_embedding.reshape(1, -1)
        similarities = np.dot(self.passage_embeddings, query_emb.T).flatten()
        s_min = float(similarities.min())
        s_max = float(similarities.max())
        norm = (
            (similarities - s_min) / (s_max - s_min)
            if s_max > s_min
            else np.zeros_like(similarities)
        )
        for p_idx, passage_id in enumerate(self.passage_ids):
            passage_text_lower = self.passage_texts[p_idx].lower()
            total_bonus = 0.0
            for entity, (_, entity_score, tier) in actived_entities.items():
                occurrences = passage_text_lower.count(entity.lower())
                if occurrences > 0:
                    denom = tier if tier >= 1 else 1
                    total_bonus += max(entity_score, 0.0) * math.log(1 + occurrences) / denom
            score = self.config.passage_ratio * float(norm[p_idx]) + math.log(
                max(1 + total_bonus, 1e-9)
            )
            node_idx = self.node_name_to_vertex_idx[passage_id]
            passage_weights[node_idx] = score * self.config.passage_node_weight
        return passage_weights

    def _run_ppr(self, node_weights: np.ndarray) -> tuple[list[str], list[float]]:
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        scores = self.graph.personalized_pagerank(
            vertices=range(len(self.graph.vs["name"])),
            damping=self.config.damping,
            directed=False,
            weights="weight",
            reset=reset_prob,
            implementation="prpack",
        )
        doc_scores = np.array([scores[i] for i in self.passage_node_indices])
        if np.isnan(doc_scores).any():
            doc_scores = np.nan_to_num(doc_scores, nan=0.0)
        order = np.argsort(doc_scores)[::-1]
        sorted_ids = [self.passage_ids[i] for i in order]
        return sorted_ids, doc_scores[order].tolist()

    def _dense_fallback(self, query_embedding: np.ndarray, top_k: int):
        query_emb = query_embedding.reshape(1, -1)
        similarities = np.dot(self.passage_embeddings, query_emb.T).flatten()
        order = np.argsort(similarities)[::-1]
        sorted_ids = [self.passage_ids[i] for i in order]
        return sorted_ids[:top_k], [float(similarities[i]) for i in order[:top_k]]

    def search(self, query: str, top_k: int = 100) -> tuple[list[str], list[float]]:
        query_embedding = np.asarray(
            self.embedder.encode(query, normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )
        seed_indices, seed_scores = self._get_seed_entities(query)
        if not seed_indices:
            return self._dense_fallback(query_embedding, top_k)
        entity_weights, actived = self._calculate_entity_scores(
            query_embedding, seed_indices, seed_scores
        )
        passage_weights = self._calculate_passage_scores(query_embedding, actived)
        node_weights = entity_weights + passage_weights
        sorted_ids, sorted_scores = self._run_ppr(node_weights)
        return sorted_ids[:top_k], sorted_scores[:top_k]
