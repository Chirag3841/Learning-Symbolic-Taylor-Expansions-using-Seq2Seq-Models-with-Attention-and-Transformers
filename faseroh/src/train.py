"""train.py — Training loop, evaluation metrics, and chi-squared GOF."""

import math
import time

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim

from .tokenizer import Vocabulary


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, dl, opt, criterion, device, is_lstm: bool,
              training: bool = True, tf: float = 0.5) -> float:
    """Run one training or evaluation epoch; return mean CE loss."""
    model.train() if training else model.eval()
    total = 0
    ctx   = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for src, tgt in dl:
            src, tgt = src.to(device), tgt.to(device)
            if is_lstm:
                out    = model(src, tgt, tf=tf if training else 0.0)
                logits = out[:, 1:].reshape(-1, out.shape[-1])
                labels = tgt[:, 1:].reshape(-1)
            else:
                out    = model(src, tgt[:, :-1])
                logits = out.reshape(-1, out.shape[-1])
                labels = tgt[:, 1:].reshape(-1)
            loss = criterion(logits, labels)
            if training:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item()
    return total / len(dl)


def fit(model, is_lstm: bool, tag: str,
        train_dl, val_dl, criterion, device,
        epochs: int = 100, save_dir: str = "/tmp") -> tuple[list, list]:
    """
    Full training with:
      - scheduled teacher-forcing decay
      - ReduceLROnPlateau
      - best-checkpoint saving

    Returns (train_losses, val_losses).
    """
    lr  = 5e-4 if is_lstm else 1e-3
    opt = optim.Adam(model.parameters(), lr=lr,
                     betas=(0.9, 0.999 if is_lstm else 0.98))
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5, min_lr=1e-5)

    best_loss = float("inf")
    tr_log, vl_log = [], []
    ckpt = f"{save_dir}/{tag}.pt"
    t0   = time.time()

    for ep in range(1, epochs + 1):
        tf_rate = max(0.1, 0.5 * (1 - ep / epochs))   # scheduled decay
        tr = run_epoch(model, train_dl, opt, criterion, device, is_lstm, training=True,  tf=tf_rate)
        vl = run_epoch(model, val_dl,   opt, criterion, device, is_lstm, training=False)
        sched.step(vl)
        tr_log.append(tr)
        vl_log.append(vl)
        if vl < best_loss:
            best_loss = vl
            torch.save(model.state_dict(), ckpt)
        if ep % 10 == 0:
            print(f"  ep {ep:3d}/{epochs}  tr={tr:.3f}  vl={vl:.3f}  "
                  f"tf={tf_rate:.2f}  ppl={math.exp(vl):.1f}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"  done in {time.time() - t0:.0f}s  best_val={best_loss:.4f}")
    return tr_log, vl_log


# ── Evaluation metrics ────────────────────────────────────────────────────────

def token_accuracy(model, dl, device, is_lstm: bool, pad_idx: int = 0) -> float:
    """Fraction of non-PAD tokens predicted correctly."""
    model.eval()
    ok = n = 0
    with torch.no_grad():
        for src, tgt in dl:
            src, tgt = src.to(device), tgt.to(device)
            if is_lstm:
                out  = model(src, tgt, tf=0.0)
                pred = out[:, 1:].argmax(-1)
                ref  = tgt[:, 1:]
            else:
                out  = model(src, tgt[:, :-1])
                pred = out.argmax(-1)
                ref  = tgt[:, 1:]
            mask = ref != pad_idx
            ok  += (pred[mask] == ref[mask]).sum().item()
            n   += mask.sum().item()
    return ok / n if n else 0.0


def seq_accuracy(model, dl, device, vocab: Vocabulary, use_beam: bool = True) -> float:
    """
    Fraction of sequences that are symbolically equivalent to the reference.
    Falls back to a 10-char prefix match on parse failure.
    """
    model.eval()
    ok = n = 0

    with torch.no_grad():
        for src, tgt in dl:
            src, tgt = src.to(device), tgt.to(device)
            for i in range(tgt.size(0)):
                src_i = src[i].unsqueeze(0)
                if use_beam and hasattr(model, "generate_beam"):
                    pred = model.generate_beam(src_i)
                else:
                    pred = model.generate(src_i)

                ref_str = vocab.decode(tgt[i].tolist())
                hyp_str = vocab.decode(pred[0].tolist())

                ref_clean = ref_str.replace(" ", "")
                hyp_clean = hyp_str.replace(" ", "").replace("**+", "**")

                try:
                    ref_sym = sp.simplify(sp.sympify(ref_clean))
                    hyp_sym = sp.simplify(sp.sympify(hyp_clean))
                    if sp.simplify(ref_sym - hyp_sym) == 0:
                        ok += 1
                    else:
                        ok += int(ref_clean[:10] == hyp_clean[:10])
                except Exception:
                    ok += int(ref_clean[:10] == hyp_clean[:10])
                n += 1

    return ok / n if n else 0.0


# ── Chi-squared goodness-of-fit ───────────────────────────────────────────────

def chi2_gof(observed, expected, min_count: int = 5) -> tuple[float | None, int | None]:
    """
    Section 2.3: X = Σ (N_k − n_k)² / n_k.
    Bins with expected count < min_count are excluded.
    Returns (X, ndf) or (None, None) if no valid bins.
    """
    Nk   = np.array(observed, dtype=float)
    nk   = np.array(expected,  dtype=float)
    mask = nk >= min_count
    if not mask.any():
        return None, None
    X   = float(np.sum((Nk[mask] - nk[mask]) ** 2 / nk[mask]))
    ndf = int(mask.sum())
    return X, ndf
