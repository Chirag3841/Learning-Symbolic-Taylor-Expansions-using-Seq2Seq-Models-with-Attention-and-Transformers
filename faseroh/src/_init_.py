"""faseroh — symbolic Taylor expansion via seq2seq learning."""

from .data      import build_taylor_data, build_histogram_data, random_pdf, make_histogram
from .tokenizer import Vocabulary, tokenize
from .dataset   import TaylorDS, make_dataloaders
from .models    import LSTMSeq2Seq, TransformerSeq2Seq, PositionalEncoding
from .train     import run_epoch, fit, token_accuracy, seq_accuracy, chi2_gof
from .plot      import plot_results

__all__ = [
    "build_taylor_data", "build_histogram_data", "random_pdf", "make_histogram",
    "Vocabulary", "tokenize",
    "TaylorDS", "make_dataloaders",
    "LSTMSeq2Seq", "TransformerSeq2Seq", "PositionalEncoding",
    "run_epoch", "fit", "token_accuracy", "seq_accuracy", "chi2_gof",
    "plot_results",
]
