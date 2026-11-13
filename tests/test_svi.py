import numpy as np
from volsurface.svi import svi_total_variance, calibrate_svi

def test_svi_calibration_on_synthetic_smile():
    k=np.linspace(-.35,.35,31); truth=(.035,.12,-.35,.02,.18)
    w=svi_total_variance(k,*truth); p=calibrate_svi(k,w,n_starts=4)
    fit=svi_total_variance(k,p["a"],p["b"],p["rho"],p["m"],p["eta"])
    assert np.sqrt(np.mean((fit-w)**2)) < 1e-4
    assert -1 < p["rho"] < 1 and p["b"] > 0 and p["eta"] > 0
