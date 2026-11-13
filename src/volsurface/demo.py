"""Explicitly synthetic EURO STOXX 50-like options for offline reproducibility."""
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
from .cleaning import black76_price


def synthetic_chain(seed=2026):
    rng=np.random.default_rng(seed)
    spot=5000.0
    rows=[]
    for days in [14,30,60,90,180,270,365,540]:
        t=days/365.25
        forward=spot*np.exp(.012*t)
        discount=np.exp(-.025*t)
        expiry=(datetime(2026,11,13,tzinfo=timezone.utc)+timedelta(days=days)).date().isoformat()
        for k in np.linspace(-.38,.32,37):
            strike=round(forward*np.exp(k),2)
            theta=(.19**2)*t
            iv=np.sqrt(theta/t)*(1-.28*k+1.6*k*k)*(1+.12*np.exp(-t*8))
            typ='put' if k<0 else 'call'
            mid=black76_price(forward,strike,t,iv,typ,discount)
            spread=max(.6,mid*(.012+.055*abs(k)))
            bid=max(mid-spread/2,.01)
            ask=mid+spread/2
            rows.append(dict(instrument_name=f'SYNTH-SX5E-{expiry}-{strike}-{typ}',
                expiry_date=expiry, strike=strike, forward=forward, maturity=t,
                option_type=typ,bid=bid,ask=ask,discount_factor=discount,
                open_interest=int(max(1,400*np.exp(-12*abs(k))+rng.normal(0,10))),
                source='SYNTHETIC_DEMO_NOT_MARKET_DATA'))
    return pd.DataFrame(rows)
