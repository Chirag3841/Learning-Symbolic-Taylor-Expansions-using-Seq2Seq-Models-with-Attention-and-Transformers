# src/plot.py
"""
Plotting utilities for FASEROH training results.

Public API
----------
plot_results(lstm_tr, lstm_vl, tf_tr, tf_vl, hist_data, out_path)
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .train import chi2_gof


def plot_results(
    lstm_tr: list[float],
    lstm_vl: list[float],
    tf_tr:   list[float],
    tf_vl:   list[float],
    hist_data: list[dict],
    out_path: str = "faseroh_results.png",
) -> None:
    """
    Save a 3-panel figure:
      [0] Training / validation loss curves for LSTM and Transformer
      [1] Validation perplexity curves
      [2] χ² / ndf distribution over the histogram dataset

    Parameters
    ----------
    lstm_tr / lstm_vl  : per-epoch train / val losses for the LSTM
    tf_tr   / tf_vl    : per-epoch train / val losses for the Transformer
    hist_data          : list of {"hist", "expected", "K", "N"} dicts
    out_path           : file path for the saved PNG
    """
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # ---- panel 0: loss curves ---------------------------------------- #
    ax = axs[0]
    ep_lstm = range(1, len(lstm_tr) + 1)
    ep_tf   = range(1, len(tf_tr)   + 1)

    ax.plot(ep_lstm, lstm_tr, "#3b82f6", lw=1.5, label="LSTM train")
    ax.plot(ep_lstm, lstm_vl, "#3b82f6", lw=1.5, ls="--", label="LSTM val")
    ax.plot(ep_tf,   tf_tr,   "#f97316", lw=1.5, label="TF train")
    ax.plot(ep_tf,   tf_vl,   "#f97316", lw=1.5, ls="--", label="TF val")
    ax.set(xlabel="epoch", ylabel="CE loss", title="Loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # ---- panel 1: perplexity ----------------------------------------- #
    ax = axs[1]
    ax.plot(ep_lstm, [math.exp(l) for l in lstm_vl], "#3b82f6", lw=1.5, label="LSTM")
    ax.plot(ep_tf,   [math.exp(l) for l in tf_vl],   "#f97316", lw=1.5, label="TF")
    ax.set(xlabel="epoch", ylabel="perplexity", title="Val perplexity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # ---- panel 2: χ² goodness-of-fit --------------------------------- #
    ax = axs[2]
    ratios = []
    for hd in hist_data:
        X, ndf = chi2_gof(hd["hist"], hd["expected"])
        if X is not None and ndf and ndf > 0:
            ratios.append(X / ndf)

    ax.hist(ratios, bins=20, color="#64748b", edgecolor="white", alpha=0.85)
    ax.axvline(1.0, color="red", ls="--", lw=1.2, label="ideal = 1")
    ax.set(xlabel="X²/ndf", ylabel="count", title="χ² goodness-of-fit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")
