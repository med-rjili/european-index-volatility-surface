import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from volsurface.svi import fit_svi_by_expiry, svi_total_variance
from volsurface.ssvi import fit_ssvi, ssvi_total_variance
from volsurface.diagnostics import error_metrics, calendar_monotonicity_check, butterfly_g_diagnostic
from volsurface.plotting import plot_market_smiles, plot_market_surface, plot_svi_fits, plot_svi_surface, plot_residual_heatmap

def main():
    p=argparse.ArgumentParser(description="Calibrate SVI/SSVI and generate figures.")
    p.add_argument("--input",required=True); p.add_argument("--output-dir",default="reports"); p.add_argument("--equal-weights",action="store_true"); a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); df=pd.read_csv(a.input)
    required={"expiry_date","time_to_maturity","log_moneyness","iv","total_variance"}
    if required-set(df): raise SystemExit(f"Missing columns: {sorted(required-set(df))}")
    plot_market_smiles(df,out/"market_smiles.png"); plot_market_surface(df,out/"market_surface.html")
    params,fitted=fit_svi_by_expiry(df,weighted=not a.equal_weights); params.to_csv(out/"svi_parameters.csv",index=False)
    diagnostics={}
    if not fitted.empty:
        plot_svi_fits(fitted,out/"svi_fits.png"); plot_svi_surface(df,params,out/"svi_surface.html"); plot_residual_heatmap(fitted,out/"residual_heatmap.png")
        metrics=error_metrics(fitted.iv,fitted.model_iv); pd.DataFrame([metrics]).to_csv(out/"calibration_metrics.csv",index=False)
    if len(params)>=2:
        kg=np.linspace(df.log_moneyness.quantile(.05),df.log_moneyness.quantile(.95),100); rows=params.sort_values("maturity").to_dict("records")
        def slice_fn(k,t):
            r=min(rows,key=lambda x:abs(x["maturity"]-t)); return svi_total_variance(k,r["a"],r["b"],r["rho"],r["m"],r["eta"])
        diagnostics["svi_calendar"]=calendar_monotonicity_check(slice_fn,params.maturity,kg)
        diagnostics["svi_butterfly"]={r["expiry_date"]:butterfly_g_diagnostic(lambda k,t,r=r:svi_total_variance(k,r["a"],r["b"],r["rho"],r["m"],r["eta"]),kg,r["maturity"]) for r in rows}
    try:
        sp,theta,sf=fit_ssvi(df); (out/"ssvi_parameters.json").write_text(json.dumps(sp,indent=2)); theta.to_csv(out/"ssvi_theta_by_maturity.csv",index=False)
        diagnostics["ssvi_fit"]=error_metrics(sf.iv,sf.ssvi_iv); kg=np.linspace(df.log_moneyness.quantile(.05),df.log_moneyness.quantile(.95),100)
        def ssvi_fn(k,t):
            th=np.interp(t,theta.maturity,theta.theta_atm); return ssvi_total_variance(k,th,sp["rho"],sp["eta"],sp["gamma"])
        diagnostics["ssvi_calendar"]=calendar_monotonicity_check(ssvi_fn,theta.maturity,kg)
        diagnostics["ssvi_butterfly"]={str(t):butterfly_g_diagnostic(ssvi_fn,kg,float(t)) for t in theta.maturity}
    except ValueError as exc: diagnostics["ssvi_note"]=str(exc)
    (out/"arbitrage_diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    print(f"Analysis complete: {out.resolve()}")
if __name__=="__main__": main()
