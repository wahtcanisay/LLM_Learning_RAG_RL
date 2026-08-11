"""LinearGraphRetriever: relation-free graph retrieval, a faithful port of
LinearRAG's default non-vectorized retrieval core, adapted to the frozen
``pubmedqa_hard_v1`` chunk data and the shared evaluation contract.

Offline build (one pass): medical NER over chunk content -> entities;
sentence split + sentence embeddings -> Entity<->Sentence bridge;
Entity-Passage edges (normalized co-occurrence) -> igraph with Entity and
Passage nodes -> persisted graph + stores + hashed report.

Online search (per query): question NER -> seed entities (argmax over corpus
entity embeddings) -> Entity->Sentence->Entity BFS propagation (semantic
bridge) -> passage prior (normalized Dense + activated-entity reward) ->
Personalized PageRank -> top-k passage (chunk) ids.
"""
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from medical_graphrag.data.io import sha256_file, write_json, write_jsonl
from medical_graphrag.retrieval.bm25 import validate_frozen_dataset

DEFAULT_EMBEDDING_MODEL = "models/all-mpnet-base-v2"
DEFAULT_NER_MODEL = "en_ner_bc5cdr_md"
TEXT_MODE = "abstract_only"


class GraphConfig:
    """Retrieval hyper-parameters (aligned to LinearRAG official defaults).

    Values now match ``LinearRAG/src/config.py`` defaults as used by the
    official ``run.py`` (which passes ``passage_ratio=2`` via CLI and leaves
    ``damping`` / ``passage_node_weight`` at their config defaults).
    """

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


def build_graph_index(
    dataset_dir: Path,
    output_dir: Path,
    *,
    config: GraphConfig | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run medical NER + sentence bridge + Entity-Passage edges, build and
    persist the igraph, and return a hashed report.

    Adjacent-passage edges are intentionally NOT added (v1): PubMedQA abstracts
    are short and global ordering would create cross-document wrong edges.
    """
    import faiss
    import igraph as ig

    if config is None:
        config = GraphConfig()
    manifest = validate_frozen_dataset(dataset_dir)
    chunks = [
        json.loads(line)
        for line in (dataset_dir / "chunks.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    if len(chunks) != manifest["counts"]["chunks"]:
        raise ValueError("chunk count does not match manifest")
    passage_ids = [str(row["chunk_id"]) for row in chunks]
    texts = [str(row["content"]) for row in chunks]

    nlp = _load_nlp(config.ner_model)
    passage_entities = extract_entities(nlp, texts, batch_size=batch_size)
    sentences = split_sentences(nlp, texts)
    del nlp

    embedder = _load_embedder(config.embedding_model)

    # Sentence bridge: map entity <-> sentences, and embed sentences.
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
            # Only entities that actually appear in this sentence bridge it
            # (closer to LinearRAG's per-sentence NER than whole-passage entities).
            sentence_entities = [e for e in passage_ents if e in sentence]
            sentence_to_entities.append(sentence_entities)
            for entity in sentence_entities:
                entity_to_sentences[entity].append(sid)

    sentence_embeddings = np.asarray(
        embedder.encode(
            sentence_texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ),
        dtype="float32",
    )
    passage_embeddings = np.asarray(
        embedder.encode(
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ),
        dtype="float32",
    )

    # Entity-Passage edges + entity list.
    all_entities, edges = build_entity_passage_edges(passage_ids, texts, passage_entities)

    # Entity embeddings (for seed-entity matching).
    entity_embeddings = np.asarray(
        embedder.encode(
            all_entities,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ),
        dtype="float32",
    )

    # igraph: Entity nodes then Passage nodes.
    graph = ig.Graph(directed=False)
    entity_index = {e: i for i, e in enumerate(all_entities)}
    passage_index = {p: len(all_entities) + i for i, p in enumerate(passage_ids)}
    graph.add_vertices(len(all_entities) + len(passage_ids))
    graph.vs["name"] = list(all_entities) + passage_ids
    graph.vs["content"] = [e for e in all_entities] + texts
    edge_list: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    for passage_id, entity_scores in edges.items():
        for entity, weight in entity_scores.items():
            edge_list.append((passage_index[passage_id], entity_index[entity]))
            edge_weights.append(weight)
    if edge_list:
        graph.add_edges(edge_list)
        graph.es["weight"] = edge_weights
    passage_node_indices = [passage_index[p] for p in passage_ids]

    # Persist.
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
    report = {
        "text_mode": TEXT_MODE,
        "ner_model": config.ner_model,
        "embedding_model": config.embedding_model,
        "entity_count": len(all_entities),
        "passage_count": len(passage_ids),
        "sentence_count": len(sentence_ids),
        "edge_count": len(edge_list),
        "passage_entity_edge_mode": "normalized_cooccurrence",
        "adjacent_passage_edges": False,
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

    def __init__(self, index_dir: Path, *, config: GraphConfig | None = None):
        import igraph as ig

        if config is None:
            config = GraphConfig()
        self.config = config
        self.index_dir = Path(index_dir)
        self.report = json.loads(
            (self.index_dir / "graph_build.json").read_text(encoding="utf-8")
        )
        self.embedder = _load_embedder(config.embedding_model)
        self.nlp = _load_nlp(config.ner_model)
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
                    # Clamp the entity score so negative seed similarities cannot
                    # drive the log argument out of its domain.
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
