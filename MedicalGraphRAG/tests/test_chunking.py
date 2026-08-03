import pytest

from medical_graphrag.data.chunking import chunk_sections


class CharacterCodec:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(value) for value in ids)


def test_chunk_sections_never_mix_sections_or_documents() -> None:
    chunks = chunk_sections(
        doc_id="PMID:1",
        title="Title",
        sections=("abcdefgh", "XYZ"),
        source="pubmedqa",
        tokenizer=CharacterCodec(),
        max_tokens=5,
        overlap=2,
    )
    assert [chunk.chunk_id for chunk in chunks] == ["PMID:1#0", "PMID:1#1", "PMID:1#2"]
    assert [chunk.content for chunk in chunks] == ["abcde", "defgh", "XYZ"]
    assert all(chunk.doc_id == "PMID:1" for chunk in chunks)


def test_chunk_sections_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        chunk_sections(
            doc_id="PMID:1",
            title="Title",
            sections=("abc",),
            source="pubmedqa",
            tokenizer=CharacterCodec(),
            max_tokens=5,
            overlap=5,
        )
