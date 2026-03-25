"""dataset.py — PyTorch Dataset and DataLoader helpers."""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from .tokenizer import Vocabulary


class TaylorDS(Dataset):
    """Map-style dataset wrapping a subset of taylor_data by index list."""

    def __init__(self, data: list[dict], indices: list[int], vocab: Vocabulary):
        self.data  = data
        self.idx   = indices
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        d   = self.data[self.idx[i]]
        src = torch.tensor(self.vocab.encode(d["src"]), dtype=torch.long)
        tgt = torch.tensor(
            [self.vocab.SOS] + self.vocab.encode(d["tgt"]) + [self.vocab.EOS],
            dtype=torch.long,
        )
        return src, tgt


def collate_fn(batch, pad_value: int = 0):
    srcs, tgts = zip(*batch)
    return (
        pad_sequence(srcs, batch_first=True, padding_value=pad_value),
        pad_sequence(tgts, batch_first=True, padding_value=pad_value),
    )


def make_dataloaders(
    data: list[dict],
    vocab: Vocabulary,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, list, list, list]:
    """
    Split data and return (train_dl, val_dl, test_dl, tr_idx, va_idx, te_idx).
    """
    all_idx = list(range(len(data)))
    tr_idx, tmp    = train_test_split(all_idx, test_size=test_size,  random_state=seed)
    va_idx, te_idx = train_test_split(tmp,     test_size=val_ratio,  random_state=seed)

    pad = vocab.PAD
    cfn = lambda b: collate_fn(b, pad_value=pad)

    train_dl = DataLoader(TaylorDS(data, tr_idx, vocab), batch_size=batch_size, shuffle=True,  collate_fn=cfn)
    val_dl   = DataLoader(TaylorDS(data, va_idx, vocab), batch_size=batch_size,                collate_fn=cfn)
    test_dl  = DataLoader(TaylorDS(data, te_idx, vocab), batch_size=batch_size,                collate_fn=cfn)

    return train_dl, val_dl, test_dl, tr_idx, va_idx, te_idx
