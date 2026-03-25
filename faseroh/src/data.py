"""data.py — Taylor series dataset and histogram PDF generation."""

import random
import sympy as sp
import numpy as np
from scipy import integrate as sci

x = sp.Symbol("x")

# ── Taylor series symbolic functions ─────────────────────────────────────────

TAYLOR_FUNCS = [
    sp.sin(x), sp.cos(x), sp.tan(x),
    sp.exp(x), sp.exp(-x),
    sp.log(1 + x), sp.log(1 + x**2),
    1 / (1 + x), 1 / (1 + x**2), 1 / (1 - x + x**2),
    sp.sqrt(1 + x),
    x**2 + x + 1, x**3 - x, (1 + x)**3, x**4 - x**2,
    sp.sinh(x), sp.cosh(x), sp.tanh(x), sp.atan(x),
    sp.sin(x) * sp.exp(x), sp.sin(x) + sp.cos(x),
    sp.exp(-x) * sp.cos(x), sp.sin(x) * sp.cos(x),
    sp.log(1 + x) * sp.cos(x), x**2 * sp.sin(x),
    sp.sin(x**2), (1 + x)**4, 1 / (1 + x + x**2),
]

COEFFS = [
    1,
    sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 4),
    sp.Rational(2, 3), sp.Rational(3, 4), sp.Rational(3, 2),
    2, 3, sp.Rational(5, 2),
]


def build_taylor_data(n_combos: int = 3000, seed: int = 99) -> list[dict]:
    """
    Build a list of {src, tgt} dicts for Taylor-series seq2seq training.

    Each entry is a symbolic function (src) paired with its degree-4
    Maclaurin expansion (tgt).  Includes single-function entries for
    every (func, coeff) pair and `n_combos` random two-function linear
    combinations.
    """
    data = []

    # single-function pairs
    for f in TAYLOR_FUNCS:
        for c in COEFFS:
            try:
                expansion = sp.expand(sp.series(c * f, x, 0, 5).removeO())
                if expansion == 0 or expansion.is_number:
                    continue
                data.append({"src": str(c * f), "tgt": str(sp.expand(expansion))})
            except Exception:
                pass

    # random combinations
    rng = random.Random(seed)
    for _ in range(n_combos):
        f1, f2 = rng.sample(TAYLOR_FUNCS, 2)
        c1, c2 = rng.choice(COEFFS), rng.choice(COEFFS)
        try:
            combo = c1 * f1 + c2 * f2
            expansion = sp.expand(sp.series(combo, x, 0, 5).removeO())
            if expansion == 0 or expansion.is_number:
                continue
            data.append({"src": str(combo), "tgt": str(sp.expand(expansion))})
        except Exception:
            pass

    return data


# ── Histogram / PDF dataset ───────────────────────────────────────────────────

_SYM_BASES = [
    1 + x,          1 + x**2,       1 + x + x**2,
    x * (1 - x),    (1 - x)**2,     x**2 * (1 - x),
    sp.exp(x),      sp.exp(-x),     sp.exp(-2 * x),   sp.exp(x) * (1 - x),
    sp.sin(sp.pi * x),
    sp.sin(2 * sp.pi * x) + 1,
    sp.cos(sp.pi * x) + 1,
    1 / (1 + x),    1 / (1 + x**2),
    sp.sqrt(x) * (1 - x),
    x**3 + 1,       sp.log(1 + x) + 1,
]


def _build_base_pdfs() -> list:
    """Compile and normalise symbolic bases into callable PDFs."""
    pdfs = []
    for sym in _SYM_BASES:
        try:
            fn = sp.lambdify(x, sym, "numpy")
            I, _ = sci.quad(fn, 0, 1)
            if I > 1e-8 and np.isfinite(I):
                pdfs.append(lambda t, fn=fn, I=I: fn(t) / I)
        except Exception:
            pass
    return pdfs


BASE_PDFS = _build_base_pdfs()


def random_pdf(base_pdfs=None):
    """
    Algorithm 2: combine 1–3 base PDFs with {+, *, /} and renormalise.
    Returns a callable f: [0,1] -> R≥0  or None on failure.
    """
    if base_pdfs is None:
        base_pdfs = BASE_PDFS

    f = random.choice(base_pdfs)
    for _ in range(random.randint(0, 2)):
        b  = random.choice(base_pdfs)
        op = random.choice(["+", "*", "/"])
        w  = random.uniform(0.5, 2.0)
        if   op == "+": combined = lambda t, f=f, b=b, w=w: f(t) + w * b(t)
        elif op == "*": combined = lambda t, f=f, b=b:       f(t) * b(t)
        else:           combined = lambda t, f=f, b=b:       f(t) / (b(t) + 0.1)
        try:
            I, _ = sci.quad(combined, 0, 1)
            if I > 1e-8 and np.isfinite(I):
                f = lambda t, g=combined, I=I: g(t) / I
            else:
                return None
        except Exception:
            return None
    return f


def make_histogram(f, N: int, K: int):
    """
    Algorithm 1: integrate f over each of K bins, draw N multinomial counts.

    Returns
    -------
    hist : np.ndarray of shape (K,)  — observed counts
    expected : np.ndarray of shape (K,)  — expected counts
    """
    edges = np.linspace(0, 1, K + 1)
    try:
        probs = np.array([
            max(sci.quad(f, edges[i], edges[i + 1])[0], 0.0)
            for i in range(K)
        ])
    except Exception:
        return None, None
    if probs.sum() < 1e-8:
        return None, None
    probs /= probs.sum()
    return np.random.multinomial(N, probs), N * probs


def build_histogram_data(n_samples: int = 200, max_attempts: int = 600) -> list[dict]:
    """Build `n_samples` histogram records with random PDFs."""
    data = []
    for _ in range(max_attempts):
        if len(data) >= n_samples:
            break
        f = random_pdf()
        if f is None:
            continue
        K = random.randint(10, 30)
        N = random.randint(200, 1000)
        h, e = make_histogram(f, N, K)
        if h is not None:
            data.append({"hist": h, "expected": e, "K": K, "N": N})
    return data
