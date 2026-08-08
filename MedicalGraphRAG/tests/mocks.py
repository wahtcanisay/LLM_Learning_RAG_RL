"""Deterministic mocks for embedder/NER so unit tests never load real models.

The MockEmbedder produces unit-normalized vectors from character hashes; the
MockNlp extracts a small fixed entity vocabulary and splits on newlines. All
vectors and entities are a pure function of the input text, so tests are
deterministic and reproducible.
"""
import numpy as np


class MockTokenizer:
    def encode(self, text, add_special_tokens=True):
        tokens = list(text)
        if add_special_tokens:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(
            token
            for token in token_ids
            if not (skip_special_tokens and token.startswith("["))
        )


class MockEmbedder:
    def __init__(self, max_seq_length: int = 512):
        self.tokenizer = MockTokenizer()
        self._max_seq_length = max_seq_length

    def get_max_seq_length(self):
        return self._max_seq_length

    def encode(self, texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        vectors = []
        for text in texts:
            vector = np.zeros(8, dtype=np.float32)
            for ch in text:
                vector[ord(ch) % 8] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            vectors.append(vector)
        result = np.array(vectors, dtype=np.float32)
        # 与 SentenceTransformer 一致:单字符串返回 1D,列表返回 2D
        return result[0] if single else result

    def encode_token_windows(self, windows, *, batch_size, normalize_embeddings):
        vectors = []
        for window in windows:
            vector = np.zeros(8, dtype=np.float32)
            for token in window:
                vector[ord(token) % 8] += 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


_ENTITY_VOCAB = ("Aspirin", "vaccine", "vaccines", "pain", "osteoarthritis", "fever")


class _MockEnt:
    def __init__(self, text):
        self.text = text
        self.label_ = "CHEMICAL"


class _MockSent:
    def __init__(self, text):
        self.text = text


class _MockDoc:
    def __init__(self, text):
        lower = text.lower()
        self.ents = [
            _MockEnt(entity) for entity in _ENTITY_VOCAB if entity.lower() in lower
        ]
        self.sents = [
            _MockSent(part) for part in text.split("\n") if part.strip()
        ] or [_MockSent(text)]


class MockNlp:
    def pipe(self, texts, batch_size=64):
        for text in texts:
            yield _MockDoc(text)

    def __call__(self, text):
        return _MockDoc(text)
