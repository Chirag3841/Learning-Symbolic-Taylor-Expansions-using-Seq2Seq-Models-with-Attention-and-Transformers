# src/tokenizer.py
"""
Tokeniser and vocabulary for symbolic math expressions.

Public API
----------
build_vocab(taylor_data)  → sets module-level t2i, i2t, vocab, PAD, SOS, EOS
tokenize(expr)            → list[str]
encode(s)                 → list[int]
decode(ids)               → str
"""

import re

# --------------------------------------------------------------------------- #
#  Module-level state (populated by build_vocab)
# --------------------------------------------------------------------------- #
SPECIAL = ["<pad>", "<sos>", "<eos>", "<unk>"]

vocab: list[str] = []
t2i:  dict[str, int] = {}
i2t:  dict[int, str] = {}
PAD = SOS = EOS = 0          # will be overwritten by build_vocab


# --------------------------------------------------------------------------- #
#  Core functions
# --------------------------------------------------------------------------- #

def tokenize(expr: str) -> list[str]:
    """Split a symbolic expression into atomic tokens."""
    return re.findall(r"\d+/\d+|\d+|\*\*|[a-zA-Z_]+|[+\-*/(),]", expr)


def build_vocab(taylor_data: list[dict]) -> dict[str, int]:
    """
    Construct vocabulary from a list of {"src", "tgt"} dicts and set the
    module-level look-up tables + special-token indices.

    Returns t2i mapping.
    """
    global vocab, t2i, i2t, PAD, SOS, EOS

    tokens: set[str] = set()
    for d in taylor_data:
        tokens.update(tokenize(d["src"]))
        tokens.update(tokenize(d["tgt"]))

    vocab = SPECIAL + sorted(tokens)
    t2i   = {tok: i for i, tok in enumerate(vocab)}
    i2t   = {i: tok for tok, i in t2i.items()}
    PAD, SOS, EOS = t2i["<pad>"], t2i["<sos>"], t2i["<eos>"]

    print(f"vocab size: {len(vocab)}")
    return t2i


def encode(s: str) -> list[int]:
    """Convert an expression string to a list of token ids."""
    return [t2i.get(tok, t2i["<unk>"]) for tok in tokenize(s)]


def decode(ids) -> str:
    """Convert token ids back to a human-readable expression string."""
    out = []
    for i in ids:
        tok = i2t.get(int(i), "")
        if tok == "<eos>":
            break
        if tok not in ("<pad>", "<sos>"):
            out.append(tok)
    return " ".join(out).replace(" / ", "/")
