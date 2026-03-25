"""plot.py — Training curves and chi-squared GOF histogram."""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .train import chi2_gof


def plot_results(
    lstm_tr, lstm_vl,
    tf_tr,   tf_vl,
    hist_data: list[dict],
    save_path: str = "faseroh_results.png",
    dpi: int = 150,
) -> None:
    """
    Three-panel figure:
      1. Training / validation CE loss for both models
      2. Validation perplexity
      3. χ² / ndf distribution over the histogram dataset
    """
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # ── panel 1: loss curves ──────────────────────────────────────────────────
    ep_lstm = range(1, len(lstm_tr) + 1)
    ep_tf   = range(1, len(tf_tr)   + 1)
    axs[0].plot(ep_lstm, lstm_tr, "#3b82f6", lw=1.5, label="LSTM train")
    axs[0].plot(ep_lstm, lstm_vl, "#3b82f6", lw=1.5, ls="--", label="LSTM val")
    axs[0].plot(ep_tf,   tf_tr,   "#f97316", lw=1.5, label="TF train")
    axs[0].plot(ep_tf,   tf_vl,   "#f97316", lw=1.5, ls="--", label="TF val")
    axs[0].set(xlabel="epoch", ylabel="CE loss", title="loss")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.2)

    # ── panel 2: perplexity ───────────────────────────────────────────────────
    axs[1].plot(ep_lstm, [math.exp(l) for l in lstm_vl], "#3b82f6", lw=1.5, label="LSTM")
    axs[1].plot(ep_tf,   [math.exp(l) for l in tf_vl],   "#f97316", lw=1.5, label="TF")
    axs[1].set(xlabel="epoch", ylabel="perplexity", title="val perplexity")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.2)

    # ── panel 3: χ²/ndf distribution ─────────────────────────────────────────
    ratios = []
    for hd in hist_data:
        X, ndf = chi2_gof(hd["hist"], hd["expected"])
        if X and ndf and ndf > 0:
            ratios.append(X / ndf)
    axs[2].hist(ratios, bins=20, color="#64748b", edgecolor="white", alpha=0.85)
    axs[2].axvline(1.0, color="red", ls="--", lw=1.2, label="ideal=1")
    axs[2].set(xlabel="X/ndf", ylabel="count", title="χ² goodness-of-fit")
    axs[2].legend(fontsize=8)
    axs[2].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"saved {save_path}")
