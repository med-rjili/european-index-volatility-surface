"""Vendor-neutral option chain validation and Black-76 implied volatility."""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


def black76_price(forward, strike, maturity, sigma, option_type='call', discount=1.0):
    if min(forward, strike, maturity, sigma, discount) <= 0:
        raise ValueError('Black-76 requires positive forward, strike, maturity, sigma and discount')
    d1 = (np.log(forward / strike) + .5*sigma*sigma*maturity)/(sigma*np.sqrt(maturity))
    d2 = d1 - sigma*np.sqrt(maturity)
    if option_type.lower() == 'call':
        return discount*(forward*norm.cdf(d1)-strike*norm.cdf(d2))
    if option_type.lower() == 'put':
        return discount*(strike*norm.cdf(-d2)-forward*norm.cdf(-d1))
    raise ValueError('option_type must be call or put')


def implied_volatility(price, forward, strike, maturity, option_type='call', discount=1.0):
    if not np.all(np.isfinite([price, forward, strike, maturity, discount])):
        return np.nan
    if min(forward, strike, maturity, discount) <= 0 or price <= 0:
        return np.nan
    intrinsic = discount*max(forward-strike, 0) if option_type.lower() == 'call' else discount*max(strike-forward, 0)
    upper = discount*(forward if option_type.lower() == 'call' else strike)
    if price <= intrinsic+1e-10 or price >= upper:
        return np.nan
    try:
        return brentq(lambda vol: black76_price(forward,strike,maturity,vol,option_type,discount)-price,
                      1e-5, 5.0, xtol=1e-11)
    except ValueError:
        return np.nan


def prepare_options(raw, min_days=3, max_relative_spread=.75):
    required = {'strike','forward','maturity','option_type','bid','ask','discount_factor'}
    missing = required-set(raw.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    df = raw.copy()
    for col in ['strike','forward','maturity','bid','ask','discount_factor','open_interest']:
        if col in df: df[col] = pd.to_numeric(df[col],errors='coerce')
    df['option_type'] = df['option_type'].astype(str).str.lower()
    valid = ((df['strike']>0)&(df['forward']>0)&(df['maturity']>=min_days/365.25)&
             (df['discount_factor']>0)&(df['discount_factor']<=1.5)&
             (df['bid']>0)&(df['ask']>=df['bid'])&df['option_type'].isin(['call','put']))
    df=df.loc[valid].copy()
    df['mid']=(df['bid']+df['ask'])/2
    df['relative_spread']=(df['ask']-df['bid'])/df['mid']
    df=df.loc[df['relative_spread']<=max_relative_spread].copy()
    df['iv']=[implied_volatility(r.mid,r.forward,r.strike,r.maturity,r.option_type,r.discount_factor)
              for r in df.itertuples()]
    df=df.loc[df['iv'].between(.01,3)].copy()
    df['time_to_maturity']=df['maturity']
    df['forward_proxy']=df['forward']
    df['log_moneyness']=np.log(df['strike']/df['forward'])
    df['total_variance']=df['iv']**2*df['maturity']
    df['expiry_date']=df['expiry_date'].astype(str) if 'expiry_date' in df else df['maturity'].round(5).astype(str)
    oi=df['open_interest'].fillna(0).clip(lower=0) if 'open_interest' in df else pd.Series(0.,index=df.index)
    df['calibration_weight']=(np.sqrt(oi+1)/(1+10*df['relative_spread'])).clip(lower=.05)
    return df.reset_index(drop=True)
