#!/usr/bin/env python3
"""scripts/train_all.py — End-to-end training run for FASEROH."""

import math
import warnings
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore")

# reproducibility
random.seed(42); np.random.seed(42); torch.manual_seed(42)

from faseroh import (
    build_taylor_data, build_histogram_data,
    Vocabulary, make_dataloaders,
    LSTMSeq2Seq, TransformerSeq2Seq,
    run_epoch, fit, token_accuracy, seq_accuracy, chi2_gof,
    plot_results,
)
from faseroh.tokenizer import tokenize


def main():
    # ── config ──────────────────────────────────────────────────────────────
    LSTM_EP  = 80
    TF_EP    = 150
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── data ─────────────────────────────────────────────────────────────────
    print("Building Taylor dataset...")
    taylor_data = build_taylor_data(n_combos=3000)
    print(f"  Taylor pairs: {len(taylor_data)}")

    print("Building histogram dataset...")
    hist_data = build_histogram_data(n_samples=200)
    print(f"  Histogram samples: {len(hist_data)}")

    # ── vocab + dataloaders ──────────────────────────────────────────────────
    vocab = Vocabulary(taylor_data)
    print(f"Vocab size: {len(vocab)}")

    train_dl, val_dl, test_dl, tr_idx, va_idx, te_idx = make_dataloaders(
        taylor_data, vocab, batch_size=32
    )
    print(f"train: {len(tr_idx)}  val: {len(va_idx)}  test: {len(te_idx)}")

    # ── chi-squared sanity check ─────────────────────────────────────────────
    print("\nchi-squared GOF check (5 samples):")
    for hd in hist_data[:5]:
        X, ndf = chi2_gof(hd["hist"], hd["expected"])
        if X and ndf:
            label = "good" if 0.5 < X / ndf < 1.5 else "check"
            print(f"  X={X:.2f}  ndf={ndf}  X/ndf={X/ndf:.3f}  ({label})")

    # ── models ───────────────────────────────────────────────────────────────
    VSIZ      = len(vocab)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD)
    print(f"\ndevice={device}  vocab={VSIZ}  lstm_epochs={LSTM_EP}  tf_epochs={TF_EP}\n")

    print("[LSTM]")
    lstm = LSTMSeq2Seq(VSIZ, pad_idx=vocab.PAD).to(device)
    lstm_tr, lstm_vl = fit(lstm, is_lstm=True, tag="lstm",
                           train_dl=train_dl, val_dl=val_dl,
                           criterion=criterion, device=device, epochs=LSTM_EP)

    print("\n[Transformer]")
    tf_model = TransformerSeq2Seq(VSIZ, d=384, heads=12, pad_idx=vocab.PAD).to(device)
    tf_tr, tf_vl = fit(tf_model, is_lstm=False, tag="tf",
                       train_dl=train_dl, val_dl=val_dl,
                       criterion=criterion, device=device, epochs=TF_EP)

    # ── results table ────────────────────────────────────────────────────────
    print(f"\n{'model':14} {'split':5} {'loss':>7} {'ppl':>8} {'tok%':>7} {'seq%':>7}")
    print("-" * 52)
    for split, dl in [("val", val_dl), ("test", test_dl)]:
        for model, is_lstm, name in [(lstm, True, "LSTM"), (tf_model, False, "Transformer")]:
            loss = run_epoch(model, dl, None, criterion, device, is_lstm, training=False)
            tok  = token_accuracy(model, dl, device, is_lstm, pad_idx=vocab.PAD)
            seq  = seq_accuracy(model, dl, device, vocab, use_beam=True)
            print(f"{name:14} {split:5} {loss:7.4f} {math.exp(loss):8.2f} "
                  f"{tok*100:6.1f}% {seq*100:6.1f}%")

    # ── sample predictions ───────────────────────────────────────────────────
    import sympy as sp
    print("\nsample predictions (5 test examples):")
    _ex  = [taylor_data[i] for i in te_idx[:5]]
    _src = pad_sequence(
        [torch.tensor(vocab.encode(d["src"])) for d in _ex],
        batch_first=True, padding_value=vocab.PAD,
    ).to(device)

    for model, name in [(lstm, "LSTM"), (tf_model, "Transformer")]:
        preds = []
        for i in range(_src.size(0)):
            src_i = _src[i].unsqueeze(0)
            pred  = (model.generate_beam(src_i) if hasattr(model, "generate_beam")
                     else model.generate(src_i))
            preds.append(pred[0])
        print(f"\n  {name}:")
        for i, d in enumerate(_ex):
            hyp  = vocab.decode(preds[i].tolist())
            try:
                diff = sp.simplify(sp.sympify(hyp) - sp.sympify(d["tgt"]))
                tick = "✓" if diff == 0 else "✗"
            except Exception:
                tick = "?"
            print(f"    [{tick}] {d['src']}")
            print(f"         ref: {d['tgt']}")
            print(f"         got: {hyp}")

    # ── plots ────────────────────────────────────────────────────────────────
    plot_results(lstm_tr, lstm_vl, tf_tr, tf_vl, hist_data)


if __name__ == "__main__":
    main()
