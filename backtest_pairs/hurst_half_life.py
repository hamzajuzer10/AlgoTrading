import numpy as np
from numpy import log, polyfit, sqrt, std, subtract
import statsmodels.api as sm

def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts.values[lag:], ts.values[:-lag]))) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)
    # Return the Hurst exponent from the polyfit output
    return round(poly[0]*2.0, 3)

def half_life(ts):
    """Returns the half life of the time series"""
    spread_lag = ts.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    spread_ret = ts - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    spread_lag2 = sm.add_constant(spread_lag)
    model = sm.OLS(spread_ret, spread_lag2)
    res = model.fit()
    halflife = round(-np.log(2) / res.params[1], 3)

    return halflife