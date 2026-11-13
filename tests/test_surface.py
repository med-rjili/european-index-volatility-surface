import numpy as np
from volsurface.surface import total_variance, implied_vol_from_total_variance

def test_total_variance_round_trip():
    iv=np.array([.4,.8,1.2]); t=np.array([.1,.5,1.])
    assert np.allclose(implied_vol_from_total_variance(total_variance(iv,t),t),iv)
