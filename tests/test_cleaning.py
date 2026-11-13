import numpy as np
from volsurface.cleaning import black76_price, implied_volatility, prepare_options
from volsurface.demo import synthetic_chain

def test_black76_iv_roundtrip():
    for typ in ('call','put'):
        price=black76_price(5000,4800,.5,.23,typ,.98)
        assert abs(implied_volatility(price,5000,4800,.5,typ,.98)-.23)<1e-8

def test_synthetic_chain_preparation():
    clean=prepare_options(synthetic_chain())
    assert len(clean)>150
    assert np.isfinite(clean['iv']).all()
    assert clean['expiry_date'].nunique()>=6
