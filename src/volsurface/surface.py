"""Volatility surface coordinate transformations."""
import numpy as np

def total_variance(iv, maturity):
    return np.asarray(iv, dtype=float) ** 2 * np.asarray(maturity, dtype=float)

def implied_vol_from_total_variance(w, maturity):
    return np.sqrt(np.maximum(np.asarray(w, dtype=float), 0) / np.asarray(maturity, dtype=float))
