"""Fit metrics and finite-grid static-arbitrage warning diagnostics."""
import numpy as np

def error_metrics(market_iv, model_iv):
    x, y = np.asarray(market_iv, float), np.asarray(model_iv, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any(): return {"n": 0, "rmse": float("nan"), "mae": float("nan")}
    e = x[mask] - y[mask]
    return {"n": int(mask.sum()), "rmse": float(np.sqrt(np.mean(e**2))), "mae": float(np.mean(np.abs(e)))}

def calendar_monotonicity_check(surface_fn, maturities, k_grid, tolerance=1e-6):
    ts, ks, violations = np.sort(np.asarray(maturities, float)), np.asarray(k_grid, float), []
    for t1, t2 in zip(ts[:-1], ts[1:]):
        diff = np.asarray(surface_fn(ks, t2)) - np.asarray(surface_fn(ks, t1))
        if np.any(diff < -tolerance):
            violations.append({"T1": float(t1), "T2": float(t2), "count": int((diff < -tolerance).sum()), "minimum_increment": float(diff.min())})
    return {"passed_on_grid": not violations, "violations": violations, "scope": "finite-grid check, not formal proof"}

def butterfly_g_diagnostic(total_variance_fn, k_grid, maturity, eps=1e-4):
    k = np.asarray(k_grid, float)
    w = np.maximum(np.asarray(total_variance_fn(k, maturity), float), 1e-12)
    wp = (total_variance_fn(k + eps, maturity) - total_variance_fn(k - eps, maturity)) / (2 * eps)
    wpp = (total_variance_fn(k + eps, maturity) - 2 * total_variance_fn(k, maturity) + total_variance_fn(k - eps, maturity)) / eps**2
    g = (1 - k * wp / (2 * w))**2 - .25 * wp**2 * (1 / w + .25) + .5 * wpp
    return {"minimum_g": float(np.nanmin(g)), "nonnegative_on_grid": bool(np.nanmin(g) >= -1e-5), "scope": "finite-grid numerical diagnostic"}
