"""Parsimonious SSVI surface calibration with power-law phi(theta)."""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

def ssvi_total_variance(k, theta, rho, eta, gamma):
    k, theta = np.asarray(k, float), np.asarray(theta, float)
    phi = eta * np.maximum(theta, 1e-10) ** (-gamma)
    return .5 * theta * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2))

def fit_ssvi(df):
    if df.empty:
        raise ValueError("Cannot fit SSVI to empty data.")
    atm = (df.assign(abs_k=df.log_moneyness.abs()).sort_values("abs_k")
           .groupby("expiry_date", as_index=False).first().sort_values("time_to_maturity"))
    if len(atm) < 2:
        raise ValueError("SSVI requires at least two maturities.")
    maturities, theta = atm.time_to_maturity.to_numpy(), atm.total_variance.to_numpy()
    T, k, w = df.time_to_maturity.to_numpy(), df.log_moneyness.to_numpy(), df.total_variance.to_numpy()
    weights = np.sqrt(np.maximum(df.get("calibration_weight", pd.Series(1., index=df.index)).to_numpy(), 1e-8))
    weights /= weights.mean()
    theta_obs = np.interp(T, maturities, theta)
    def residual(p):
        return weights * (ssvi_total_variance(k, theta_obs, *p) - w)
    best = None
    for start in ([0., .5, .3], [-.3, .8, .2], [.3, .3, .5]):
        res = least_squares(residual, start, bounds=([-.95, 1e-4, .01], [.95, 10., .99]))
        score = float(np.sum(res.fun ** 2))
        if best is None or score < best[0]: best = (score, res)
    score, res = best
    rho, eta, gamma = res.x
    out = df.copy()
    out["theta_at_maturity"] = theta_obs
    out["ssvi_total_variance"] = ssvi_total_variance(k, theta_obs, rho, eta, gamma)
    out["ssvi_iv"] = np.sqrt(np.maximum(out.ssvi_total_variance, 0) / T)
    out["ssvi_iv_residual"] = out.iv - out.ssvi_iv
    return {"rho": float(rho), "eta": float(eta), "gamma": float(gamma), "weighted_sse": score, "success": bool(res.success)}, pd.DataFrame({"maturity": maturities, "theta_atm": theta}), out
