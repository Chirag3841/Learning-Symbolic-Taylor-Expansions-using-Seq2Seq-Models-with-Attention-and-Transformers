#!/usr/bin/env python3
# scripts/train_all.py
"""
End-to-end training entry point for FASEROH.

Usage
-----
    python scripts/train_all.py [--lstm-epochs N] [--tf-epochs N] [--out results.png]

Steps
-----
1. Build Taylor dataset
2. Build histogram dataset
3. Build vocabulary
4. Create DataLoaders
5. Train LSTM seq2seq
6. Train Transformer seq2seq
7. Evaluate both models (token accuracy + sequence accuracy)
8. Print sample predictions
9. Save result plot
"""

import argparse
import math
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings("ignore")

# ---- reproducibility ----------------------------------------------------- #
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---- project imports ----------------------------------------------------- #
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.data      import build_taylor_data, build_histogram_data
from src.tokenizer import build_vocab, encode, decode, PAD
from src.dataset   import make_dataloaders
from src.models    import LSTMSeq2Seq, TransformerSeq2Seq
from src.train     import (
    fit, run_epoch, token_accuracy, seq_accuracy, safe_parse, chi2_gof
)
from src.plot      import plot_results


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="FASEROH training pipeline")
    p.add_argument("--lstm-epochs", type=int, default=40,
                   help="number of epochs for the LSTM model (default: 40)")
    p.add_argument("--tf-epochs",   type=int, default=80,
                   help="number of epochs for the Transformer (default: 80)")
    p.add_argument("--out", type=str, default="faseroh_results.png",
                   help="output path for the results plot")
    p.add_argument("--checkpoint-dir", type=str, default="/tmp",
                   help="directory to save model checkpoints")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    # ---- 1. Data --------------------------------------------------------- #
    taylor_data = build_taylor_data()
    hist_data   = build_histogram_data(n_samples=200)

    # ---- 2. Vocabulary --------------------------------------------------- #
    build_vocab(taylor_data)

    # ---- 3. DataLoaders -------------------------------------------------- #
    train_dl, val_dl, test_dl, tr_idx, va_idx, te_idx = make_dataloaders(
        taylor_data, batch_size=32
    )

    # ---- 4. Device & loss ------------------------------------------------ #
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VSIZ      = len(__import__("src.tokenizer", fromlist=["vocab"]).vocab)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    print(f"\ndevice={device}  vocab={VSIZ}  "
          f"lstm_epochs={args.lstm_epochs}  tf_epochs={args.tf_epochs}\n")

    # ---- 5. LSTM --------------------------------------------------------- #
    print("[LSTM]")
    lstm = LSTMSeq2Seq(VSIZ).to(device)
    lstm_tr, lstm_vl = fit(
        lstm, train_dl, val_dl, criterion, device,
        is_lstm=True, tag="lstm",
        epochs=args.lstm_epochs,
        checkpoint_dir=args.checkpoint_dir,
    )

    # ---- 6. Transformer -------------------------------------------------- #
    print("\n[Transformer]")
    tf_model = TransformerSeq2Seq(VSIZ, d=384, heads=12).to(device)
    tf_tr, tf_vl = fit(
        tf_model, train_dl, val_dl, criterion, device,
        is_lstm=False, tag="tf",
        epochs=args.tf_epochs,
        checkpoint_dir=args.checkpoint_dir,
    )

    # ---- 7. Evaluation table --------------------------------------------- #
    print(f"\n{'model':14} {'split':5} {'loss':>7} {'ppl':>8} {'tok%':>7} {'seq%':>7}")
    print("-" * 52)

    for split, dl in [("val", val_dl), ("test", test_dl)]:
        for model, is_lstm, name in [
            (lstm,     True,  "LSTM"),
            (tf_model, False, "Transformer"),
        ]:
            loss = run_epoch(model, dl, None, criterion, device, is_lstm, training=False)
            tok  = token_accuracy(model, dl, device, is_lstm) * 100
            seq  = seq_accuracy(model, dl, device, use_beam=True) * 100
            print(f"{name:14} {split:5} {loss:7.4f} {math.exp(loss):8.2f} "
                  f"{tok:6.1f}% {seq:6.1f}%")

    # ---- 8. Sample predictions ------------------------------------------- #
    import src.tokenizer as tok_module
    _ex  = [taylor_data[i] for i in te_idx[:5]]
    _src = pad_sequence(
        [torch.tensor(encode(d["src"])) for d in _ex],
        batch_first=True, padding_value=PAD,
    ).to(device)

    print("\nsample predictions (5 test examples):")
    for model, name in [(lstm, "LSTM"), (tf_model, "Transformer")]:
        preds = []
        for i in range(_src.size(0)):
            src_i = _src[i].unsqueeze(0)
            pred  = (model.generate_beam(src_i)
                     if hasattr(model, "generate_beam")
                     else model.generate(src_i))
            preds.append(pred[0])

        print(f"\n  {name}:")
        for i, d in enumerate(_ex):
            hyp = decode(preds[i].tolist())
            try:
                diff = safe_parse(hyp) - safe_parse(d["tgt"])
                tick = "✓" if __import__("sympy").simplify(diff) == 0 else "✗"
            except Exception:
                tick = "✗"
            print(f"    [{tick}] {d['src']}")
            print(f"         ref: {d['tgt']}")
            print(f"         got: {hyp}")

    # ---- 9. Plot --------------------------------------------------------- #
    plot_results(lstm_tr, lstm_vl, tf_tr, tf_vl, hist_data, out_path=args.out)


if __name__ == "__main__":
    main()
