"""tokenizer.py — Tokenisation and vocabulary for symbolic expressions."""

import re

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]


def tokenize(expr: str) -> list[str]:
    """Split a SymPy string expression into atomic tokens."""
    return re.findall(r'\d+|\*\*|[a-zA-Z]+|[+\-*/()]', expr)


class Vocabulary:
    """
    Character-level vocabulary built from a list of (src, tgt) expression pairs.

    Attributes
    ----------
    tokens : list[str]   — ordered token list (special first, then sorted)
    t2i    : dict        — token -> index
    i2t    : dict        — index -> token
    PAD, SOS, EOS, UNK : int  — indices of special tokens
    """

    def __init__(self, data: list[dict]):
        raw: set[str] = set()
        for d in data:
            raw.update(tokenize(d["src"]))
            raw.update(tokenize(d["tgt"]))

        self.tokens = SPECIAL_TOKENS + sorted(raw)
        self.t2i = {tok: i for i, tok in enumerate(self.tokens)}
        self.i2t = {i: tok for tok, i in self.t2i.items()}

        self.PAD = self.t2i["<pad>"]
        self.SOS = self.t2i["<sos>"]
        self.EOS = self.t2i["<eos>"]
        self.UNK = self.t2i["<unk>"]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, s: str) -> list[int]:
        return [self.t2i.get(tok, self.UNK) for tok in tokenize(s)]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            tok = self.i2t.get(int(i), "")
            if tok == "<eos>":
                break
            if tok not in ("<pad>", "<sos>"):
                out.append(tok)
        return " ".join(out)
