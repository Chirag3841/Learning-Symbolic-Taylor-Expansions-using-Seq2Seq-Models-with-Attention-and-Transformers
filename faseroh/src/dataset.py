# src/dataset.py
"""
PyTorch Dataset and DataLoader helpers for the Taylor-series task.

Public API
----------
TaylorDS          – torch.utils.data.Dataset subclass
make_dataloaders  – returns (train_dl, val_dl, test_dl, tr_idx, va_idx, te_idx)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from .tokenizer import encode, PAD, SOS, EOS


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #

class TaylorDS(Dataset):
    """
    Wraps a list of {"src": str, "tgt": str} dicts.
    Selects entries by index list so train/val/test share one backing store.
    """

    def __init__(self, taylor_data: list[dict], indices: list[int]):
        self.data = taylor_data
        self.idx  = indices

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        d   = self.data[self.idx[i]]
        src = torch.tensor(encode(d["src"]), dtype=torch.long)
        tgt = torch.tensor([SOS] + encode(d["tgt"]) + [EOS], dtype=torch.long)
        return src, tgt


# --------------------------------------------------------------------------- #
#  Collate + DataLoaders
# --------------------------------------------------------------------------- #

def _collate(batch):
    srcs, tgts = zip(*batch)
    return (
        pad_sequence(srcs, batch_first=True, padding_value=PAD),
        pad_sequence(tgts, batch_first=True, padding_value=PAD),
    )


def make_dataloaders(
    taylor_data: list[dict],
    batch_size: int = 32,
    test_size: float = 0.2,
    val_fraction: float = 0.5,
    random_state: int = 42,
    num_workers: int = 0,
):
    """
    Split *taylor_data* into train / val / test and return DataLoaders.

    Returns
    -------
    train_dl, val_dl, test_dl, tr_idx, va_idx, te_idx
    """
    all_idx = list(range(len(taylor_data)))
    tr_idx, tmp   = train_test_split(all_idx,  test_size=test_size,    random_state=random_state)
    va_idx, te_idx = train_test_split(tmp,      test_size=val_fraction, random_state=random_state)

    print(f"train: {len(tr_idx)}  val: {len(va_idx)}  test: {len(te_idx)}")

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=_collate,
        pin_memory=True,
        num_workers=num_workers,
    )

    train_dl = DataLoader(TaylorDS(taylor_data, tr_idx), shuffle=True,  **loader_kwargs)
    val_dl   = DataLoader(TaylorDS(taylor_data, va_idx), shuffle=False, **loader_kwargs)
    test_dl  = DataLoader(TaylorDS(taylor_data, te_idx), shuffle=False, **loader_kwargs)

    return train_dl, val_dl, test_dl, tr_idx, va_idx, te_idx
