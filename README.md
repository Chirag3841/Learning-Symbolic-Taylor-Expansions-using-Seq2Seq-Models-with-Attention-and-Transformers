# Learning-Symbolic-Taylor-Expansions-using-Seq2Seq-Models-with-Attention-and-Transformers
# FASEROH
Fast Accurate Symbolic Empirical Representation Of Histograms

##  Overview

This project explores neural approaches to symbolic mathematical reasoning by training sequence-to-sequence (seq2seq) models to generate degree-4 Maclaurin (Taylor) expansions of symbolic expressions.
Additionally, the project validates a synthetic histogram dataset using a χ² (chi-squared) goodness-of-fit test to ensure statistical correctness.
The core idea is to treat symbolic mathematics as a translation problem:
function → Taylor expansion

---

##  Motivation

Seq2seq models have shown strong performance in neural machine translation. This project applies similar ideas to symbolic mathematics, where:
- exact precision is important  
- structural understanding is critical  
- neural models struggle with numerical accuracy  
This work highlights both the strengths and limitations of neural symbolic learning.

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
├── faseroh_project.py # (place Colab/Jupyter notebooks here)
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
<img width="2083" height="581" alt="faseroh_results (4)" src="https://github.com/user-attachments/assets/9caa45b7-edb7-4d09-97da-8d95ed5c9790" />



**Left:** CE loss curves — the Transformer converges faster and generalises better (val loss ≈ 0.60 vs LSTM ≈ 1.33). The LSTM train loss drops near zero but val loss stays elevated, indicating overfitting. **Centre:** Val perplexity follows the same story — Transformer stabilises at ~1.8× vs LSTM at ~6×. **Right:** The χ²/ndf distribution is centred just below 1, confirming the multinomial histogram samples are well-calibrated against their generating PDFs.

| model | split | loss | ppl | tok acc | seq acc |
|---|---|---|---|---|---|
| LSTM | val | 1.3275 | 3.77 | 71.7% | 9.8% |
| Transformer | val | 0.5946 | 1.81 | 81.6% | 1.5% |
| LSTM | test | 1.5701 | 4.81 | 70.1% | 7.9% |
| Transformer | test | 0.6043 | 1.83 | 81.1% | 1.8% |

## Observations
Transformer achieves higher token accuracy (~80%)
LSTM achieves higher sequence accuracy (~8–10%)
Models learn correct polynomial structure
Errors mainly occur in numerical coefficients

 ## Key Insight

Neural models capture symbolic structure effectively but struggle with precise numerical reasoning.

##  Visualizations
Loss curves show stable convergence (Transformer better)
Perplexity indicates better generalization
χ²/ndf ≈ 1 confirms dataset validity

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
