"""Static and interactive visualizations for market and fitted volatility surfaces."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from .svi import svi_total_variance

def plot_market_smiles(df, output):
    fig, ax = plt.subplots(figsize=(10, 6))
    for expiry, g in df.groupby("expiry_date"):
        g = g.sort_values("log_moneyness")
        ax.plot(g.log_moneyness, g.iv * 100, marker=".", lw=1, label=expiry)
    ax.set(title="European equity index options — implied-volatility smiles", xlabel="Log-forward-moneyness", ylabel="Black–76 implied volatility (%)")
    ax.grid(alpha=.25); ax.legend(title="Expiry", fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)

def plot_market_surface(df, output_html):
    fig = go.Figure(data=[go.Scatter3d(x=df.log_moneyness, y=df.time_to_maturity, z=df.iv*100, mode="markers",
        marker=dict(size=3, color=df.iv*100, colorscale="Viridis", colorbar=dict(title="IV (%)")),
        text=df.instrument_name if "instrument_name" in df else None)])
    fig.update_layout(title="European equity index options — observed implied volatilities", scene=dict(xaxis_title="Log-forward-moneyness", yaxis_title="Time to maturity (years)", zaxis_title="Implied volatility (%)"))
    fig.write_html(str(output_html), include_plotlyjs="cdn")

def plot_svi_fits(fitted, output):
    if fitted.empty: return
    fig, ax = plt.subplots(figsize=(10, 6))
    for expiry, g in fitted.groupby("expiry_date"):
        g = g.sort_values("log_moneyness")
        ax.scatter(g.log_moneyness, g.iv*100, s=12, alpha=.65)
        ax.plot(g.log_moneyness, g.model_iv*100, lw=1.5, label=expiry)
    ax.set(title="Market observations and raw-SVI fitted smiles", xlabel="Log-forward-moneyness", ylabel="Implied volatility (%)")
    ax.grid(alpha=.25); ax.legend(title="Expiry", fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)

def plot_svi_surface(df, params, output_html):
    if params.empty: return
    kg = np.linspace(df.log_moneyness.quantile(.02), df.log_moneyness.quantile(.98), 50)
    tg = np.sort(params.maturity.unique()); K, T = np.meshgrid(kg, tg); Z = np.full_like(K, np.nan)
    for j, row in params.reset_index(drop=True).iterrows():
        w = svi_total_variance(kg, row.a, row.b, row.rho, row.m, row.eta)
        Z[j] = np.sqrt(np.maximum(w, 0)/row.maturity)*100
    fig = go.Figure(data=[go.Surface(x=K, y=T, z=Z, colorscale="Viridis")])
    fig.update_layout(title="Raw-SVI calibrated volatility surface", scene=dict(xaxis_title="Log-forward-moneyness", yaxis_title="Maturity (years)", zaxis_title="IV (%)"))
    fig.write_html(str(output_html), include_plotlyjs="cdn")

def plot_residual_heatmap(fitted, output):
    if fitted.empty: return
    pivot = fitted.pivot_table(index="expiry_date", columns="log_moneyness", values="iv_residual", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 5)); im = ax.imshow(pivot.to_numpy()*100, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set(title="Raw-SVI residuals (market IV − model IV)", xlabel="Sorted log-forward-moneyness observations", ylabel="Expiry")
    ax.set_yticks(range(len(pivot.index)), labels=pivot.index); fig.colorbar(im, ax=ax, label="IV residual (volatility points)")
    fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)
