from MedSearch.rl.retriever_client import encode_observation


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]


def test_truncation_keeps_information_closed():
    tokenizer = CharTokenizer()
    observation = "<information>" + "x" * 100 + "</information>"
    token_ids = encode_observation(observation, tokenizer, 40)
    assert len(token_ids) == 40
    assert "".join(map(chr, token_ids)).endswith("</information>")
    assert encode_observation("<information>x</information>", tokenizer, 40) == tokenizer.encode(
        "<information>x</information>"
    )
