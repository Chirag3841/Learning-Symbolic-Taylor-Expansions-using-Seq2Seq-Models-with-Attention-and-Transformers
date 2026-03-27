## FASEROH — Symbolic Taylor-Series Regression

Seq2seq models (LSTM + Transformer) that learn to expand symbolic expressions
into their 4th-order Taylor series around x = 0.

## Project layout
faseroh/
├── src/
│   ├── __init__.py      # public API re-exports
│   ├── data.py          # Taylor dataset + histogram PDF generation
│   ├── tokenizer.py     # tokenise / vocabulary
│   ├── dataset.py       # PyTorch Dataset & DataLoader helpers
│   ├── models.py        # LSTMSeq2Seq, TransformerSeq2Seq
│   ├── train.py         # training loop, metrics, chi-squared GOF
│   └── plot.py          # result figures
├── scripts/
│   └── train_all.py     # end-to-end training entry point
├── pyproject.toml
├── requirements.txt
└── README.md

## Quick start
pip install -r requirements.txt
python scripts/train_all.py --lstm-epochs 40 --tf-epochs 80

Results are saved to faseroh_results.png.
<img width="2098" height="581" alt="faseroh_results (4)" src="https://github.com/user-attachments/assets/f905dfc3-1db4-4fcb-8a22-04371ea5d9fd" />


## CLI options
flag	default	description
--lstm-epochs	40	training epochs for the LSTM
--tf-epochs	80	training epochs for the Transformer
--out	faseroh_results.png	output plot path
--checkpoint-dir	/tmp	directory for .pt checkpoints
Module overview
module	responsibility
src/data.py	generate Taylor pairs and histogram samples
src/tokenizer.py	character-level tokeniser + vocab
src/dataset.py	TaylorDS dataset + train/val/test split
src/models.py	LSTM (Bahdanau attention) + Transformer
src/train.py	training loop, metrics, χ² GOF
src/plot.py	result visualization

## Results
Model	Split	Loss ↓	PPL ↓	Token Acc ↑	Seq Acc ↑
LSTM	Val	0.7891	2.20	80.6%	2.5%
Transformer	Val	0.2754	1.32	92.0%	13.8%
LSTM	Test	0.8472	2.33	80.3%	3.4%
Transformer	Test	0.2922	1.34	92.3%	8.0%
Training logs (summary)
device=cuda  vocab=155  lstm_epochs=40  tf_epochs=80

[LSTM]
  ep 10/40  tr=0.877  vl=1.284  tf=0.25  ppl=3.6
  ep 20/40  tr=0.452  vl=1.015  tf=0.10  ppl=2.8
  ep 30/40  tr=0.182  vl=0.908  tf=0.10  ppl=2.5
  ep 40/40  tr=0.058  vl=0.936  tf=0.10  ppl=2.5
  done in 208s  best_val=0.7891

[Transformer]
  ep 10/80  tr=0.493  vl=0.482  tf=0.38  ppl=1.6
  ep 20/80  tr=0.307  vl=0.355  tf=0.25  ppl=1.4
  ep 30/80  tr=0.218  vl=0.317  tf=0.12  ppl=1.4
  ep 40/80  tr=0.174  vl=0.305  tf=0.10  ppl=1.4
  ep 50/80  tr=0.090  vl=0.282  tf=0.10  ppl=1.3
  ep 60/80  tr=0.071  vl=0.306  tf=0.10  ppl=1.4
  ep 70/80  tr=0.041  vl=0.317  tf=0.10  ppl=1.4
  ep 80/80  tr=0.030  vl=0.316  tf=0.10  ppl=1.4
  done in 210s  best_val=0.2754
 
## Output figure panels
Loss — train/val CE loss curves
Val perplexity — per-epoch validation perplexity
χ² / ndf — histogram goodness-of-fit (ideal ≈ 1)

## Requirements
Python ≥ 3.10
PyTorch ≥ 2.2
SymPy, SciPy, scikit-learn, matplotlib

## Author
Chirag Sharma
BTech CSE — MSIT Delhi
