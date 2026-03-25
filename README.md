# Learning-Symbolic-Taylor-Expansions-using-Seq2Seq-Models-with-Attention-and-Transformers
# FASEROH
Fast Accurate Symbolic Empirical Representation Of Histograms

Trains an LSTM (with Bahdanau attention) and a Transformer to predict the
degree-4 Maclaurin expansion of a symbolic expression — and validates the
underlying histogram dataset with a chi-squared goodness-of-fit test.

---

## Repo structure

```
faseroh/
├── src
│   ├── __init__.py      # public API
│   ├── data.py          # Taylor dataset + histogram PDF generation
│   ├── tokenizer.py     # tokenise / vocabulary
│   ├── dataset.py       # PyTorch Dataset & DataLoader helpers
│   ├── models.py        # LSTMSeq2Seq, TransformerSeq2Seq
│   ├── train.py         # training loop, metrics, chi-squared GOF
│   └── plot.py          # result figures
├── scripts/
│   └── train_all.py     # end-to-end training entry point
├── notebooks/
    └──faseroh_project.py # (place Colab/Jupyter notebooks here)
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
# 1. clone & install
git clone https://github.com/<you>/faseroh.git
cd faseroh
pip install -e ".[dev]"

# 2. train
python scripts/train_all.py
```

This will:
- generate ~3 000+ Taylor-series pairs
- build 200 random-PDF histogram samples
- train LSTM (80 epochs) and Transformer (150 epochs)
- print a results table (loss / ppl / tok-acc / seq-acc)
- save `faseroh_results.png`

---

## Results

![Training curves and χ² GOF](faseroh_results.png)

**Left:** CE loss curves — the Transformer converges faster and generalises better (val loss ≈ 0.60 vs LSTM ≈ 1.33). The LSTM train loss drops near zero but val loss stays elevated, indicating overfitting. **Centre:** Val perplexity follows the same story — Transformer stabilises at ~1.8× vs LSTM at ~6×. **Right:** The χ²/ndf distribution is centred just below 1, confirming the multinomial histogram samples are well-calibrated against their generating PDFs.

| model | split | loss | ppl | tok acc | seq acc |
|---|---|---|---|---|---|
| LSTM | val | 1.3275 | 3.77 | 71.7% | 9.8% |
| Transformer | val | 0.5946 | 1.81 | 81.6% | 1.5% |
| LSTM | test | 1.5701 | 4.81 | 70.1% | 7.9% |
| Transformer | test | 0.6043 | 1.83 | 81.1% | 1.8% |

The Transformer wins on token-level accuracy by ~10 pp, but the LSTM achieves higher **sequence accuracy** (9.8% vs 1.5% on val). This is likely because the LSTM's beam search explores a more diverse set of candidate sequences given its recurrent state, while the Transformer tends to produce locally plausible but globally incorrect expansions. Both models have meaningful room for improvement — longer training, larger model size, or data augmentation are natural next steps.

---

## Key design choices

| Choice | Reason |
|---|---|
| Bahdanau attention on LSTM | Single h,c bottleneck fails on 30+ char symbolic strings |
| Bidirectional encoder | Richer context for the decoder |
| Length-normalised beam search | Prevents the model from preferring short outputs |
| Scheduled teacher-forcing decay | Bridges train/inference gap |
| ReduceLROnPlateau | Robust to noisy symbolic loss landscapes |
| χ²/ndf ≈ 1 sanity check | Validates that multinomial histogram samples match the true PDF |

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.2
- SymPy, SciPy, scikit-learn, matplotlib
