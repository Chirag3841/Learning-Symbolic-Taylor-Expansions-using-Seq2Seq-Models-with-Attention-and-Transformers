## FASEROH — Symbolic Taylor-Series Regression

Seq2seq models (LSTM + Transformer) that learn to expand symbolic expressions
into their 4th-order Taylor series around x = 0.

## Project layout
faseroh/
├── src/
│   ├── __init__.py      # public API re-exports
│   ├── data.py          # Taylor dataset + histogram PDF generation
│   ├── tokenizer.py     # tokenisation and vocabulary
│   ├── dataset.py       # PyTorch Dataset and DataLoader helpers
│   ├── models.py        # LSTMSeq2Seq and TransformerSeq2Seq
│   ├── train.py         # training loop, metrics, χ² goodness-of-fit
│   └── plot.py          # result visualisation
├── scripts/
│   └── train_all.py     # end-to-end training entry point
├── pyproject.toml       # project configuration
├── requirements.txt     # dependencies
└── README.md            # project documentation

## Quick start
pip install -r requirements.txt
python scripts/train_all.py --lstm-epochs 40 --tf-epochs 80

Results are saved to faseroh_results.png.
<img width="2098" height="581" alt="faseroh_results (4)" src="https://github.com/user-attachments/assets/f905dfc3-1db4-4fcb-8a22-04371ea5d9fd" />


## CLI options
| Flag               | Default               | Description                         |
| ------------------ | --------------------- | ----------------------------------- |
| `--lstm-epochs`    | 40                    | Training epochs for the LSTM        |
| `--tf-epochs`      | 80                    | Training epochs for the Transformer |
| `--out`            | `faseroh_results.png` | Output plot path                    |
| `--checkpoint-dir` | `/tmp`                | Directory for `.pt` checkpoints     |

## Module overview
| Module             | Responsibility                              |
| ------------------ | ------------------------------------------- |
| `src/data.py`      | Generate Taylor pairs and histogram samples |
| `src/tokenizer.py` | Character-level tokenizer and vocabulary    |
| `src/dataset.py`   | `TaylorDS` dataset and train/val/test split |
| `src/models.py`    | LSTM (Bahdanau attention) and Transformer   |
| `src/train.py`     | Training loop, metrics, χ² goodness-of-fit  |
| `src/plot.py`      | Result visualization                        |



## Results
| Model       | Split | Loss ↓ | PPL ↓ | Token Acc ↑ | Seq Acc ↑ |
| ----------- | ----- | ------ | ----- | ----------- | --------- |
| LSTM        | Val   | 0.7891 | 2.20  | 80.6%       | 2.5%      |
| Transformer | Val   | 0.2754 | 1.32  | 92.0%       | 13.8%     |
| LSTM        | Test  | 0.8472 | 2.33  | 80.3%       | 3.4%      |
| Transformer | Test  | 0.2922 | 1.34  | 92.3%       | 8.0%      |

[LSTM]
| Epoch | Train Loss | Val Loss | TF Ratio | PPL |
| ----- | ---------- | -------- | -------- | --- |
| 10/40 | 0.877      | 1.284    | 0.25     | 3.6 |
| 20/40 | 0.452      | 1.015    | 0.10     | 2.8 |
| 30/40 | 0.182      | 0.908    | 0.10     | 2.5 |
| 40/40 | 0.058      | 0.936    | 0.10     | 2.5 |

| Metric        | Value  |
| ------------- | ------ |
| Training Time | 208s   |
| Best Val Loss | 0.7891 |


[Transformer]
| Epoch | Train Loss | Val Loss | TF Ratio | PPL |
| ----- | ---------- | -------- | -------- | --- |
| 10/80 | 0.493      | 0.482    | 0.38     | 1.6 |
| 20/80 | 0.307      | 0.355    | 0.25     | 1.4 |
| 30/80 | 0.218      | 0.317    | 0.12     | 1.4 |
| 40/80 | 0.174      | 0.305    | 0.10     | 1.4 |
| 50/80 | 0.090      | 0.282    | 0.10     | 1.3 |
| 60/80 | 0.071      | 0.306    | 0.10     | 1.4 |
| 70/80 | 0.041      | 0.317    | 0.10     | 1.4 |
| 80/80 | 0.030      | 0.316    | 0.10     | 1.4 |

| Metric        | Value  |
| ------------- | ------ |
| Training Time | 210s   |
| Best Val Loss | 0.2754 |



 
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
