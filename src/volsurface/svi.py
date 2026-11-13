"""Raw SVI total-variance smile and bounded nonlinear calibration."""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

def svi_total_variance(k, a, b, rho, m, eta):
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + eta ** 2))

def calibrate_svi(k, w, weights=None, n_starts=8, random_state=7):
    k, w = np.asarray(k, float), np.asarray(w, float)
    mask = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[mask], w[mask]
    if weights is None:
        weights = np.ones_like(w)
    else:
        weights = np.maximum(np.asarray(weights, float)[mask], 1e-8)
        weights /= weights.mean()
    if len(k) < 5 or np.unique(k).size < 5:
        raise ValueError("At least five distinct valid strikes are required for SVI.")
    rng = np.random.default_rng(random_state)
    scale, kmin, kmax = max(float(w.max()), 1e-4), float(k.min()), float(k.max())
    lo = np.array([-2 * scale, 1e-8, -.999, kmin - 2, 1e-5])
    hi = np.array([max(2 * scale, .1), max(5 * scale, .5), .999, kmax + 2, 2.0])
    starts = [np.array([max(float(w.min()) * .5, 1e-6), max(scale * .25, 1e-4), 0., float(np.median(k)), .15])]
    for _ in range(max(n_starts - 1, 0)):
        starts.append(np.array([rng.uniform(max(lo[0], -.01), min(hi[0], max(float(w.min()), .05))),
                                rng.uniform(max(lo[1], 1e-4), min(hi[1], max(scale, .2))),
                                rng.uniform(-.8, .8), rng.uniform(kmin, kmax), rng.uniform(.03, .8)]))
    best = None
    for x0 in starts:
        x0 = np.minimum(np.maximum(x0, lo + 1e-10), hi - 1e-10)
        result = least_squares(lambda p: np.sqrt(weights) * (svi_total_variance(k, *p) - w),
                               x0, bounds=(lo, hi), max_nfev=20000)
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result)
    score, result = best
    a, b, rho, m, eta = result.x
    return dict(a=float(a), b=float(b), rho=float(rho), m=float(m), eta=float(eta),
                weighted_sse=score, success=bool(result.success), n_obs=len(k))

def fit_svi_by_expiry(df, weighted=True, min_points=6):
    rows, fitted = [], []
    for expiry, group in df.groupby("expiry_date"):
        group = group.sort_values("log_moneyness")
        if group.log_moneyness.nunique() < min_points:
            continue
        try:
            p = calibrate_svi(group.log_moneyness, group.total_variance,
                              group.calibration_weight if weighted else None)
        except ValueError:
            continue
        t = float(group.time_to_maturity.median())
        rows.append({"expiry_date": expiry, "maturity": t, **p})
        out = group.copy()
        out["model_total_variance"] = svi_total_variance(out.log_moneyness, p["a"], p["b"], p["rho"], p["m"], p["eta"])
        out["model_iv"] = np.sqrt(np.maximum(out.model_total_variance, 0) / t)
        out["iv_residual"] = out.iv - out.model_iv
        fitted.append(out)
    return pd.DataFrame(rows), pd.concat(fitted, ignore_index=True) if fitted else pd.DataFrame()
