# src/train.py
"""
Training loop, evaluation metrics and chi-squared goodness-of-fit.

Public API
----------
chi2_gof(observed, expected)           → (chi2_stat, ndf)
safe_parse(expr)                       → sympy expression
run_epoch(model, dl, opt, ...)         → mean CE loss
token_accuracy(model, dl, ...)         → float ∈ [0, 1]
seq_accuracy(model, dl, ...)           → float ∈ [0, 1]
fit(model, is_lstm, tag, ...)          → (tr_log, vl_log)
"""

import math
import time
from pathlib import Path

import numpy as np
import sympy as sp
import torch
import torch.nn as nn
import torch.optim as optim

from .tokenizer import PAD, encode, decode

x = sp.Symbol("x")

# --------------------------------------------------------------------------- #
#  Chi-squared goodness-of-fit
# --------------------------------------------------------------------------- #

def chi2_gof(observed, expected):
    """
    Pearson chi-squared goodness-of-fit statistic.

    Parameters
    ----------
    observed : array-like  — integer counts per bin
    expected : array-like  — expected (float) counts per bin

    Returns
    -------
    (X2, ndf) where ndf = number of bins with expected > 0
    """
    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)
    mask = exp > 0
    if mask.sum() == 0:
        return None, None
    X2  = float(np.sum((obs[mask] - exp[mask]) ** 2 / exp[mask]))
    ndf = int(mask.sum())
    return X2, ndf


# --------------------------------------------------------------------------- #
#  Symbolic utilities
# --------------------------------------------------------------------------- #

def safe_parse(expr: str):
    """Normalise and sympify an expression string."""
    expr = expr.replace(" ", "")
    expr = expr.replace("+-", "-").replace("-+", "-")
    expr = expr.replace("--", "+").replace("++", "+")
    expr = expr.replace("*+", "*").replace("+*", "*")
    expr = expr.replace("-*", "-1*")
    return sp.sympify(expr)


# --------------------------------------------------------------------------- #
#  Single-epoch forward / backward pass
# --------------------------------------------------------------------------- #

def run_epoch(
    model,
    dl,
    opt,
    criterion,
    device,
    is_lstm: bool,
    training: bool = True,
    tf: float = 0.5,
) -> float:
    """
    Run one epoch and return mean cross-entropy loss.

    Parameters
    ----------
    model    : LSTMSeq2Seq or TransformerSeq2Seq
    dl       : DataLoader
    opt      : optimizer (ignored when training=False)
    criterion: loss function
    device   : torch.device
    is_lstm  : whether the model is LSTM (different forward signature)
    training : if False, runs under torch.no_grad() without optimizer step
    tf       : teacher-forcing ratio (LSTM only)
    """
    model.train() if training else model.eval()
    total = 0.0
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
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # vital
                opt.step()

            total += loss.item()

    return total / max(1, len(dl))


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def token_accuracy(model, dl, device, is_lstm: bool) -> float:
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
            mask  = ref != PAD
            ok   += (pred[mask] == ref[mask]).sum().item()
            n    += mask.sum().item()
    return ok / n if n else 0.0


def seq_accuracy(
    model,
    dl,
    device,
    use_beam: bool = True,
) -> float:
    """
    Fraction of sequences whose decoded output is symbolically equivalent
    to the reference (exact sympy simplify OR point-wise numerical check).
    """
    model.eval()
    ok = n = 0

    with torch.no_grad():
        for src, tgt in dl:
            src, tgt = src.to(device), tgt.to(device)

            for i in range(tgt.size(0)):
                src_i = src[i].unsqueeze(0)

                if use_beam and hasattr(model, "generate_beam"):
                    pred = model.generate_beam(src_i, beam_width=7)
                else:
                    pred = model.generate(src_i)

                ref_str = decode(tgt[i].tolist())
                hyp_str = decode(pred[0].tolist())

                try:
                    ref = safe_parse(ref_str)
                    hyp = safe_parse(hyp_str)

                    if sp.simplify(ref - hyp) == 0:
                        ok += 1
                    else:
                        # numerical fallback
                        f_ref = sp.lambdify(x, ref, "numpy")
                        f_hyp = sp.lambdify(x, hyp, "numpy")
                        pts   = np.linspace(-1, 1, 5)
                        valid = True
                        for p in pts:
                            try:
                                if abs(f_ref(p) - f_hyp(p)) > 1e-3:
                                    valid = False
                                    break
                            except Exception:
                                valid = False
                                break
                        if valid:
                            ok += 1
                except Exception:
                    pass

                n += 1

    return ok / n if n else 0.0


# --------------------------------------------------------------------------- #
#  Training orchestration
# --------------------------------------------------------------------------- #

def fit(
    model,
    train_dl,
    val_dl,
    criterion,
    device,
    is_lstm: bool,
    tag: str,
    epochs: int = 100,
    checkpoint_dir: str = "/tmp",
) -> tuple[list[float], list[float]]:
    """
    Full training loop with:
      • Adam optimiser
      • teacher-forcing decay (LSTM)
      • transformer warmup (5 epochs)
      • ReduceLROnPlateau scheduler
      • gradient clipping (norm = 1.0)
      • best-model checkpointing

    Returns
    -------
    (train_losses, val_losses)  — one value per epoch
    """
    lr  = 3e-4 if is_lstm else 5e-4
    opt = optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999 if is_lstm else 0.98),
    )
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=8, factor=0.5, min_lr=1e-5
    )

    ckpt_path = Path(checkpoint_dir) / f"{tag}.pt"
    best_loss = float("inf")
    tr_log, vl_log = [], []
    t0 = time.time()

    for ep in range(1, epochs + 1):

        # teacher-forcing decay: starts at 0.5, ends at ~0.1
        tf = max(0.1, 0.5 * (1 - 2 * ep / epochs))

        # linear warmup for Transformer
        if not is_lstm and ep <= 5:
            for g in opt.param_groups:
                g["lr"] = lr * (ep / 5)

        tr = run_epoch(model, train_dl, opt, criterion, device, is_lstm,
                       training=True,  tf=tf)
        vl = run_epoch(model, val_dl,   opt, criterion, device, is_lstm,
                       training=False)

        sched.step(vl)
        tr_log.append(tr)
        vl_log.append(vl)

        # save best checkpoint
        if vl < best_loss - 1e-4:
            best_loss = vl
            torch.save(model.state_dict(), ckpt_path)

        if ep % 10 == 0:
            print(
                f"  ep {ep:>3}/{epochs}  "
                f"tr={tr:.3f}  vl={vl:.3f}  "
                f"tf={tf:.2f}  ppl={math.exp(vl):.1f}"
            )

    # restore best weights
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"  done in {time.time() - t0:.0f}s  best_val={best_loss:.4f}")

    return tr_log, vl_log
