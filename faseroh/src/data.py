# src/data.py
"""
Taylor-series dataset generation (Algorithm 1 & 2 from the FASEROH paper).
  - build_taylor_data()   → list of {"src": str, "tgt": str} pairs
  - build_histogram_data() → list of {"hist", "expected", "K", "N"} dicts
"""

import random
import numpy as np
import sympy as sp
from scipy import integrate as sci

# --------------------------------------------------------------------------- #
#  Symbolic building-blocks
# --------------------------------------------------------------------------- #
x = sp.Symbol("x")

_TAYLOR_FUNCS = [
    sp.sin(x),  sp.cos(x),
    sp.exp(x),  sp.exp(-x),
    sp.log(1 + x),
    1 / (1 + x), 1 / (1 + x**2),
    sp.sqrt(1 + x),
    x**2 + x + 1,
    x**3 - x,
    (1 + x)**3,
    x**4 - x**2,
    sp.sinh(x), sp.cosh(x), sp.tanh(x), sp.atan(x),
    sp.sin(x) + sp.cos(x),
    sp.exp(-x) * sp.cos(x),
    sp.sin(x) * sp.cos(x),
    x**2 * sp.sin(x),
    (1 + x)**4,
]

_COEFFS = [
    1,
    sp.Rational(1, 2), sp.Rational(1, 3),
    2, 3,
]

_SYM_BASES = [
    1 + x, 1 + x**2, 1 + x + x**2,
    x * (1 - x), (1 - x)**2,
    sp.exp(x), sp.exp(-x), sp.exp(-2 * x),
    sp.sin(sp.pi * x),
    sp.sin(2 * sp.pi * x) + 1,
    sp.cos(sp.pi * x) + 1,
    1 / (1 + x), 1 / (1 + x**2),
    x**3 + 1,
    sp.log(1 + x) + 1,
]


# --------------------------------------------------------------------------- #
#  Taylor dataset
# --------------------------------------------------------------------------- #

def _single_pairs(funcs, coeffs):
    """Enumerate single-function × coefficient pairs."""
    data = []
    for f in funcs:
        for c in coeffs:
            try:
                src_expr = sp.expand(c * f)
                if len(str(src_expr)) > 50:
                    continue
                tgt_expr = sp.expand(
                    sp.series(src_expr, x, 0, 5).removeO(), order="lex"
                )
                if tgt_expr == 0 or tgt_expr.is_number:
                    continue
                if len(str(tgt_expr)) > 60:
                    continue
                data.append({"src": str(src_expr), "tgt": str(tgt_expr)})
            except Exception:
                pass
    return data


def _combo_pairs(funcs, coeffs, n_attempts: int = 5000, seed: int = 99):
    """Random combinations of two base functions."""
    rng = random.Random(seed)
    data = []
    for _ in range(n_attempts):
        f1, f2 = rng.sample(funcs, 2)
        c1, c2 = rng.choice(coeffs), rng.choice(coeffs)
        try:
            combo = sp.expand(c1 * f1 + c2 * f2)
            if len(str(combo)) > 50:
                continue
            expansion = sp.expand(
                sp.series(combo, x, 0, 5).removeO(), order="lex"
            )
            if expansion == 0 or expansion.is_number:
                continue
            if len(str(expansion)) > 60:
                continue
            data.append({"src": str(combo), "tgt": str(expansion)})
        except Exception:
            pass
    return data


def build_taylor_data(
    funcs=None,
    coeffs=None,
    combo_attempts: int = 5000,
    combo_seed: int = 99,
) -> list[dict]:
    """
    Build and deduplicate the full Taylor-series dataset.

    Returns
    -------
    list of {"src": str, "tgt": str}
    """
    funcs  = funcs  or _TAYLOR_FUNCS
    coeffs = coeffs or _COEFFS

    raw = _single_pairs(funcs, coeffs) + _combo_pairs(
        funcs, coeffs, n_attempts=combo_attempts, seed=combo_seed
    )

    # stable dedup
    seen, clean = set(), []
    for d in raw:
        key = (d["src"], d["tgt"])
        if key not in seen:
            seen.add(key)
            clean.append(d)

    print(f"Taylor pairs (after dedup): {len(clean)}")
    return clean


# --------------------------------------------------------------------------- #
#  Histogram / PDF dataset
# --------------------------------------------------------------------------- #

def _build_base_pdfs(sym_bases=None) -> list:
    """Lambdify + normalise base PDFs once."""
    sym_bases = sym_bases or _SYM_BASES
    pdfs = []
    for sym in sym_bases:
        try:
            fn = sp.lambdify(x, sym, "numpy")
            I, _ = sci.quad(fn, 0, 1)
            if I > 1e-8 and np.isfinite(I):
                pdfs.append(lambda t, _fn=fn, _I=I: _fn(t) / _I)
        except Exception:
            pass
    return pdfs


def _random_pdf(base_pdfs: list):
    """Algorithm 2: combine 1–3 base PDFs with {+, *, /}, renormalise."""
    f = random.choice(base_pdfs)
    for _ in range(random.randint(0, 2)):
        b  = random.choice(base_pdfs)
        op = random.choice(["+", "*", "/"])
        w  = random.uniform(0.5, 2.0)
        if   op == "+": combined = lambda t, _f=f, _b=b, _w=w: _f(t) + _w * _b(t)
        elif op == "*": combined = lambda t, _f=f, _b=b:        _f(t) * _b(t)
        else:           combined = lambda t, _f=f, _b=b:        _f(t) / (_b(t) + 0.1)
        try:
            I, _ = sci.quad(combined, 0, 1)
            if I > 1e-8 and np.isfinite(I):
                f = lambda t, _g=combined, _I=I: _g(t) / _I
            else:
                return None
        except Exception:
            return None
    return f


def _make_histogram(f, N: int, K: int):
    """
    Algorithm 1: integrate f over each bin → multinomial draw.

    Returns (histogram array, expected-counts array) or (None, None).
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


def build_histogram_data(
    n_samples: int = 200,
    max_attempts: int = 600,
    sym_bases=None,
) -> list[dict]:
    """
    Build a list of histogram samples.

    Returns
    -------
    list of {"hist": ndarray, "expected": ndarray, "K": int, "N": int}
    """
    base_pdfs = _build_base_pdfs(sym_bases)
    print(f"base PDFs ready: {len(base_pdfs)}")

    data = []
    print("building histogram dataset...", end=" ", flush=True)
    for _ in range(max_attempts):
        if len(data) >= n_samples:
            break
        f = _random_pdf(base_pdfs)
        if f is None:
            continue
        K = random.randint(10, 30)
        N = random.randint(200, 1000)
        h, e = _make_histogram(f, N, K)
        if h is not None:
            data.append({"hist": h, "expected": e, "K": K, "N": N})

    print(f"{len(data)} samples ready")
    return data
