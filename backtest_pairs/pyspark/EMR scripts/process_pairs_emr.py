from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType, DateType, FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pyspark.sql.functions as func
import pandas as pd
from time import time
import numpy as np
import sys
import os

## Use the following guide to setup an AWS EMR cluster
# https://towardsdatascience.com/getting-started-with-pyspark-on-amazon-emr-c85154b6b921
# Use emr-5.25 as main software configuration and only keep hadoop, hive, spark and livy due to issues with Jupyter kernels
# and bootstrap script on other software configurations
# Supply the following configuration when creating a cluster:
# [{"Classification": "spark-defaults", "Properties": {"spark.driver.memory": "20G"}},{"classification": "livy-conf","Properties": {"livy.server.session.timeout":"8h"}}]

os.environ["ARROW_PRE_0_15_IPC_FORMAT"] = "1"

csv_path = "s3://algo-trading-hjuzer/etf_tickers_12_2020.csv"
save_path = "s3://algo-trading-hjuzer/etf_tickers_results_12_2020.csv"
min_period_yrs = 1.5
max_half_life = 12 # in time interval units, 30 if days, 12 if weeks
min_half_life = 2 # in time interval units, 2 is default
time_interval = 'weekly'

# Create Spark session
conf = SparkConf()
spark = SparkSession.builder.config(conf=conf) \
    .getOrCreate()

# Enable Arrow optimization and fallback if there is no Arrow installed
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")

    # Read the dataframe from a csv
schema = StructType([
    StructField('formatted_date', DateType(), True),
    StructField('x_0', FloatType(), True),
    StructField('x_1', FloatType(), True),
    StructField('combination', StringType(), True)
])

df = spark.read.csv(csv_path, header=True, schema=schema, dateFormat="dd/MM/yyyy")
df = df.withColumn('formatted_date',func.to_timestamp(func.col('formatted_date'), "yyyy-MM-dd"))

######### Johansen test code ##############

'''
function result = johansen(x,p,k)
% PURPOSE: perform Johansen cointegration tests
% -------------------------------------------------------
% USAGE: result = johansen(x,p,k)
% where:      x = input matrix of time-series in levels, (nobs x m)
%             p = order of time polynomial in the null-hypothesis
%                 p = -1, no deterministic part
%                 p =  0, for constant term
%                 p =  1, for constant plus time-trend
%                 p >  1, for higher order polynomial
%             k = number of lagged difference terms used when
%                 computing the estimator
% -------------------------------------------------------
% RETURNS: a results structure:
%          result.eig  = eigenvalues  (m x 1)
%          result.evec = eigenvectors (m x m), where first
%                        r columns are normalized coint vectors
%          result.lr1  = likelihood ratio trace statistic for r=0 to m-1
%                        (m x 1) vector
%          result.lr2  = maximum eigenvalue statistic for r=0 to m-1
%                        (m x 1) vector
%          result.cvt  = critical values for trace statistic
%                        (m x 3) vector [90% 95% 99%]
%          result.cvm  = critical values for max eigen value statistic
%                        (m x 3) vector [90% 95% 99%]
%          result.ind  = index of co-integrating variables ordered by
%                        size of the eigenvalues from large to small
% -------------------------------------------------------
% NOTE: c_sja(), c_sjt() provide critical values generated using
%       a method of MacKinnon (1994, 1996).
%       critical values are available for n<=12 and -1 <= p <= 1,
%       zeros are returned for other cases.
% -------------------------------------------------------
% SEE ALSO: prt_coint, a function that prints results
% -------------------------------------------------------
% References: Johansen (1988), 'Statistical Analysis of Co-integration
% vectors', Journal of Economic Dynamics and Control, 12, pp. 231-254.
% MacKinnon, Haug, Michelis (1996) 'Numerical distribution
% functions of likelihood ratio tests for cointegration',
% Queen's University Institute for Economic Research Discussion paper.
% (see also: MacKinnon's JBES 1994 article
% -------------------------------------------------------

% written by:
% James P. LeSage, Dept of Economics
% University of Toledo
% 2801 W. Bancroft St,
% Toledo, OH 43606
% jlesage@spatial-econometrics.com

% ****************************************************************
% NOTE: Adina Enache provided some bug fixes and corrections that
%       she notes below in comments. 4/10/2000
% ****************************************************************
'''

from numpy import zeros, ones, flipud, log
from numpy.linalg import inv, eig, cholesky as chol
from statsmodels.regression.linear_model import OLS

tdiff = np.diff

class Holder(object):
    pass

def rows(x):
    return x.shape[0]

def trimr(x, front, end):
    if end > 0:
        return x[front:-end]
    else:
        return x[front:]

import statsmodels.tsa.tsatools as tsat
mlag = tsat.lagmat

def mlag_(x, maxlag):
    '''return all lags up to maxlag
    '''
    return x[:-lag]

def lag(x, lag):
    return x[:-lag]

def detrend(y, order):
    if order == -1:
        return y
    return OLS(y, np.vander(np.linspace(-1, 1, len(y)), order + 1)).fit().resid

def resid(y, x):
    r = y - np.dot(x, np.dot(np.linalg.pinv(x), y))
    return r




def coint_johansen(x, p, k, print_on_console=True):

    #    % error checking on inputs
    #    if (nargin ~= 3)
    #     error('Wrong # of inputs to johansen')
    #    end
    nobs, m = x.shape

    # why this?  f is detrend transformed series, p is detrend data
    if (p > -1):
        f = 0
    else:
        f = p

    x = detrend(x, p)
    dx = tdiff(x, 1, axis=0)
    # dx    = trimr(dx,1,0)
    z = mlag(dx, k)  # [k-1:]
#    print z.shape
    z = trimr(z, k, 0)
    z = detrend(z, f)
#    print dx.shape
    dx = trimr(dx, k, 0)

    dx = detrend(dx, f)
    # r0t   = dx - z*(z\dx)
    r0t = resid(dx, z)  # diff on lagged diffs
    # lx = trimr(lag(x,k),k,0)
    lx = lag(x, k)
    lx = trimr(lx, 1, 0)
    dx = detrend(lx, f)
#    print 'rkt', dx.shape, z.shape
    # rkt   = dx - z*(z\dx)
    rkt = resid(dx, z)  # level on lagged diffs
    skk = np.dot(rkt.T, rkt) / rows(rkt)
    sk0 = np.dot(rkt.T, r0t) / rows(rkt)
    s00 = np.dot(r0t.T, r0t) / rows(r0t)
    sig = np.dot(sk0, np.dot(inv(s00), (sk0.T)))
    tmp = inv(skk)
    # du, au = eig(np.dot(tmp, sig))
    au, du = eig(np.dot(tmp, sig))  # au is eval, du is evec
    # orig = np.dot(tmp, sig)

    # % Normalize the eigen vectors such that (du'skk*du) = I
    temp = inv(chol(np.dot(du.T, np.dot(skk, du))))
    dt = np.dot(du, temp)


    # JP: the next part can be done much  easier

    # %      NOTE: At this point, the eigenvectors are aligned by column. To
    # %            physically move the column elements using the MATLAB sort,
    # %            take the transpose to put the eigenvectors across the row

    # dt = transpose(dt)

    # % sort eigenvalues and vectors

    # au, auind = np.sort(diag(au))
    auind = np.argsort(au)
    # a = flipud(au)
    aind = flipud(auind)
    a = au[aind]
    # d = dt[aind,:]
    d = dt[:, aind]

    # %NOTE: The eigenvectors have been sorted by row based on auind and moved to array "d".
    # %      Put the eigenvectors back in column format after the sort by taking the
    # %      transpose of "d". Since the eigenvectors have been physically moved, there is
    # %      no need for aind at all. To preserve existing programming, aind is reset back to
    # %      1, 2, 3, ....

    # d  =  transpose(d)
    # test = np.dot(transpose(d), np.dot(skk, d))

    # %EXPLANATION:  The MATLAB sort function sorts from low to high. The flip realigns
    # %auind to go from the largest to the smallest eigenvalue (now aind). The original procedure
    # %physically moved the rows of dt (to d) based on the alignment in aind and then used
    # %aind as a column index to address the eigenvectors from high to low. This is a double
    # %sort. If you wanted to extract the eigenvector corresponding to the largest eigenvalue by,
    # %using aind as a reference, you would get the correct eigenvector, but with sorted
    # %coefficients and, therefore, any follow-on calculation would seem to be in error.
    # %If alternative programming methods are used to evaluate the eigenvalues, e.g. Frame method
    # %followed by a root extraction on the characteristic equation, then the roots can be
    # %quickly sorted. One by one, the corresponding eigenvectors can be generated. The resultant
    # %array can be operated on using the Cholesky transformation, which enables a unit
    # %diagonalization of skk. But nowhere along the way are the coefficients within the
    # %eigenvector array ever changed. The final value of the "beta" array using either method
    # %should be the same.


    # % Compute the trace and max eigenvalue statistics */
    lr1 = zeros(m)
    lr2 = zeros(m)
    cvm = zeros((m, 3))
    cvt = zeros((m, 3))
    iota = ones(m)
    t, junk = rkt.shape
    for i in range(0, m):
        tmp = trimr(log(iota - a), i , 0)
        lr1[i] = -t * np.sum(tmp, 0)  # columnsum ?
        # tmp = np.log(1-a)
        # lr1[i] = -t * np.sum(tmp[i:])
        lr2[i] = -t * log(1 - a[i])
        cvm[i, :] = c_sja(m - i, p)
        cvt[i, :] = c_sjt(m - i, p)
        aind[i] = i
    # end

    result = Holder()
    # % set up results structure
    # estimation results, residuals
    result.rkt = rkt
    result.r0t = r0t
    result.eig = a
    result.evec = d  # transposed compared to matlab ?
    result.lr1 = lr1
    result.lr2 = lr2
    result.cvt = cvt
    result.cvm = cvm
    result.ind = aind
    result.meth = 'johansen'

    if print_on_console == True:
        print ('--------------------------------------------------')
        print ('--> Trace Statistics')
        print ('variable statistic Crit-90% Crit-95%  Crit-99%')
        for i in range(len(result.lr1)):
            print ('r =', i, '\t', round(result.lr1[i], 4), result.cvt[i, 0], result.cvt[i, 1], result.cvt[i, 2])
        print ('--------------------------------------------------')
        print ('--> Eigen Statistics')
        print ('variable statistic Crit-90% Crit-95%  Crit-99%')
        for i in range(len(result.lr2)):
            print ('r =', i, '\t', round(result.lr2[i], 4), result.cvm[i, 0], result.cvm[i, 1], result.cvm[i, 2])
        print ('--------------------------------------------------')
        print ('eigenvectors:\n', result.evec)
        print ('--------------------------------------------------')
        print ('eigenvalues:\n', result.eig)
        print ('--------------------------------------------------')


    return result

def c_sjt(n, p):

# PURPOSE: find critical values for Johansen trace statistic
# ------------------------------------------------------------
# USAGE:  jc = c_sjt(n,p)
# where:    n = dimension of the VAR system
#               NOTE: routine doesn't work for n > 12
#           p = order of time polynomial in the null-hypothesis
#                 p = -1, no deterministic part
#                 p =  0, for constant term
#                 p =  1, for constant plus time-trend
#                 p >  1  returns no critical values
# ------------------------------------------------------------
# RETURNS: a (3x1) vector of percentiles for the trace
#          statistic for [90# 95# 99#]
# ------------------------------------------------------------
# NOTES: for n > 12, the function returns a (3x1) vector of zeros.
#        The values returned by the function were generated using
#        a method described in MacKinnon (1996), using his FORTRAN
#        program johdist.f
# ------------------------------------------------------------
# SEE ALSO: johansen()
# ------------------------------------------------------------
# # References: MacKinnon, Haug, Michelis (1996) 'Numerical distribution
# functions of likelihood ratio tests for cointegration',
# Queen's University Institute for Economic Research Discussion paper.
# -------------------------------------------------------

# written by:
# James P. LeSage, Dept of Economics
# University of Toledo
# 2801 W. Bancroft St,
# Toledo, OH 43606
# jlesage@spatial-econometrics.com
#
# Ported to Python by Javier Garcia
# javier.macro.trader@gmail.com

# these are the values from Johansen's 1995 book
# for comparison to the MacKinnon values
# jcp0 = [ 2.98   4.14   7.02
#        10.35  12.21  16.16
#        21.58  24.08  29.19
#        36.58  39.71  46.00
#        55.54  59.24  66.71
#        78.30  86.36  91.12
#       104.93 109.93 119.58
#       135.16 140.74 151.70
#       169.30 175.47 187.82
#       207.21 214.07 226.95
#       248.77 256.23 270.47
#       293.83 301.95 318.14];




    jcp0 = ((2.9762, 4.1296, 6.9406),
            (10.4741, 12.3212, 16.3640),
            (21.7781, 24.2761, 29.5147),
            (37.0339, 40.1749, 46.5716),
            (56.2839, 60.0627, 67.6367),
            (79.5329, 83.9383, 92.7136),
            (106.7351, 111.7797, 121.7375),
            (137.9954, 143.6691, 154.7977),
            (173.2292, 179.5199, 191.8122),
            (212.4721, 219.4051, 232.8291),
            (255.6732, 263.2603, 277.9962),
            (302.9054, 311.1288, 326.9716))


    jcp1 = ((2.7055, 3.8415, 6.6349),
            (13.4294, 15.4943, 19.9349),
            (27.0669, 29.7961, 35.4628),
            (44.4929, 47.8545, 54.6815),
            (65.8202, 69.8189, 77.8202),
            (91.1090, 95.7542, 104.9637),
            (120.3673, 125.6185, 135.9825),
            (153.6341, 159.5290, 171.0905),
            (190.8714, 197.3772, 210.0366),
            (232.1030, 239.2468, 253.2526),
            (277.3740, 285.1402, 300.2821),
            (326.5354, 334.9795, 351.2150))

    jcp2 = ((2.7055, 3.8415, 6.6349),
            (16.1619, 18.3985, 23.1485),
            (32.0645, 35.0116, 41.0815),
            (51.6492, 55.2459, 62.5202),
            (75.1027, 79.3422, 87.7748),
            (102.4674, 107.3429, 116.9829),
            (133.7852, 139.2780, 150.0778),
            (169.0618, 175.1584, 187.1891),
            (208.3582, 215.1268, 228.2226),
            (251.6293, 259.0267, 273.3838),
            (298.8836, 306.8988, 322.4264),
            (350.1125, 358.7190, 375.3203))



    if (p > 1) or (p < -1):
        jc = (0, 0, 0)
    elif (n > 12) or (n < 1):
        jc = (0, 0, 0)
    elif p == -1:
        jc = jcp0[n - 1]
    elif p == 0:
        jc = jcp1[n - 1]
    elif p == 1:
        jc = jcp2[n - 1]



    return jc

def c_sja(n, p):

# PURPOSE: find critical values for Johansen maximum eigenvalue statistic
# ------------------------------------------------------------
# USAGE:  jc = c_sja(n,p)
# where:    n = dimension of the VAR system
#           p = order of time polynomial in the null-hypothesis
#                 p = -1, no deterministic part
#                 p =  0, for constant term
#                 p =  1, for constant plus time-trend
#                 p >  1  returns no critical values
# ------------------------------------------------------------
# RETURNS: a (3x1) vector of percentiles for the maximum eigenvalue
#          statistic for: [90# 95# 99#]
# ------------------------------------------------------------
# NOTES: for n > 12, the function returns a (3x1) vector of zeros.
#        The values returned by the function were generated using
#        a method described in MacKinnon (1996), using his FORTRAN
#        program johdist.f
# ------------------------------------------------------------
# SEE ALSO: johansen()
# ------------------------------------------------------------
# References: MacKinnon, Haug, Michelis (1996) 'Numerical distribution
# functions of likelihood ratio tests for cointegration',
# Queen's University Institute for Economic Research Discussion paper.
# -------------------------------------------------------

# written by:
# James P. LeSage, Dept of Economics
# University of Toledo
# 2801 W. Bancroft St,
# Toledo, OH 43606
# jlesage@spatial-econometrics.com
# Ported to Python by Javier Garcia
# javier.macro.trader@gmail.com


    jcp0 = ((2.9762, 4.1296, 6.9406),
            (9.4748, 11.2246, 15.0923),
            (15.7175, 17.7961, 22.2519),
            (21.8370, 24.1592, 29.0609),
            (27.9160, 30.4428, 35.7359),
            (33.9271, 36.6301, 42.2333),
            (39.9085, 42.7679, 48.6606),
            (45.8930, 48.8795, 55.0335),
            (51.8528, 54.9629, 61.3449),
            (57.7954, 61.0404, 67.6415),
            (63.7248, 67.0756, 73.8856),
            (69.6513, 73.0946, 80.0937))

    jcp1 = ((2.7055, 3.8415, 6.6349),
            (12.2971, 14.2639, 18.5200),
            (18.8928, 21.1314, 25.8650),
            (25.1236, 27.5858, 32.7172),
            (31.2379, 33.8777, 39.3693),
            (37.2786, 40.0763, 45.8662),
            (43.2947, 46.2299, 52.3069),
            (49.2855, 52.3622, 58.6634),
            (55.2412, 58.4332, 64.9960),
            (61.2041, 64.5040, 71.2525),
            (67.1307, 70.5392, 77.4877),
            (73.0563, 76.5734, 83.7105))

    jcp2 = ((2.7055, 3.8415, 6.6349),
            (15.0006, 17.1481, 21.7465),
            (21.8731, 24.2522, 29.2631),
            (28.2398, 30.8151, 36.1930),
            (34.4202, 37.1646, 42.8612),
            (40.5244, 43.4183, 49.4095),
            (46.5583, 49.5875, 55.8171),
            (52.5858, 55.7302, 62.1741),
            (58.5316, 61.8051, 68.5030),
            (64.5292, 67.9040, 74.7434),
            (70.4630, 73.9355, 81.0678),
            (76.4081, 79.9878, 87.2395))


    if (p > 1) or (p < -1):
        jc = (0, 0, 0)
    elif (n > 12) or (n < 1):
        jc = (0, 0, 0)
    elif p == -1:
        jc = jcp0[n - 1]
    elif p == 0:
        jc = jcp1[n - 1]
    elif p == 1:
        jc = jcp2[n - 1]


    return jc
#######################################################


###################Half life code######################
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

#####################################################

###################utils#############################
import os

def create_dict(ticker, max_date=None, min_date=None, n_samples=None,
                johansen_90_p_trace=None, johansen_95_p_trace=None, johansen_99_p_trace=None, johansen_trace_stat=None,
                johansen_90_p_eigen=None, johansen_95_p_eigen=None, johansen_99_p_eigen=None, johansen_eigen_stat=None,
                johansen_eigenvectors=None, johansen_eigenvalue=None, merged_prices_comb_df=None, adf_test_stat=None,
                adf_99_p_stat=None, adf_95_p_stat=None, adf_90_p_stat=None, hurst_exp=None, half_life_=None, sample_pass=False,
                comment=None, save_price_df=True):

    if save_price_df:
        data = {'ticker': ticker,
                'max_date': max_date,
                'min_date': min_date,
                'n_samples': n_samples,
                'johansen_90_p_trace': johansen_90_p_trace,
                'johansen_95_p_trace': johansen_95_p_trace,
                'johansen_99_p_trace': johansen_99_p_trace,
                'johansen_trace_stat': johansen_trace_stat,
                'johansen_90_p_eigen': johansen_90_p_eigen,
                'johansen_95_p_eigen': johansen_95_p_eigen,
                'johansen_99_p_eigen': johansen_99_p_eigen,
                'johansen_eigen_stat': johansen_eigen_stat,
                'johansen_eigenvectors': johansen_eigenvectors,
                'johansen_eigenvalue': johansen_eigenvalue,
                'merged_prices_comb_df': merged_prices_comb_df,
                'adf_test_stat': adf_test_stat,
                'adf_99_p_stat': adf_99_p_stat,
                'adf_95_p_stat': adf_95_p_stat,
                'adf_90_p_stat': adf_90_p_stat,
                'hurst_exp': hurst_exp,
                'half_life_': half_life_,
                'sample_pass': sample_pass,
                'comment': comment}
    else:
        data = {'ticker': ticker,
                'max_date': max_date,
                'min_date': min_date,
                'n_samples': n_samples,
                'johansen_90_p_trace': johansen_90_p_trace,
                'johansen_95_p_trace': johansen_95_p_trace,
                'johansen_99_p_trace': johansen_99_p_trace,
                'johansen_trace_stat': johansen_trace_stat,
                'johansen_90_p_eigen': johansen_90_p_eigen,
                'johansen_95_p_eigen': johansen_95_p_eigen,
                'johansen_99_p_eigen': johansen_99_p_eigen,
                'johansen_eigen_stat': johansen_eigen_stat,
                'johansen_eigenvectors': johansen_eigenvectors,
                'johansen_eigenvalue': johansen_eigenvalue,
                'adf_test_stat': adf_test_stat,
                'adf_99_p_stat': adf_99_p_stat,
                'adf_95_p_stat': adf_95_p_stat,
                'adf_90_p_stat': adf_90_p_stat,
                'hurst_exp': hurst_exp,
                'half_life_': half_life_,
                'sample_pass': sample_pass,
                'comment': comment}

    return data

##################################################################

#########################Co-integration test###########################
from io import StringIO
import statsmodels.tsa.stattools as ts


class StdoutRedirection:
    """Standard output redirection context manager"""

    def __init__(self, path, write_mode="w"):
        self._result = StringIO()
        self._path = path
        self._write_mode = write_mode

    def __enter__(self):
        sys.stdout = self._result
        # sys.stdout = open(self._path, mode=self._write_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # sys.stdout.close()
        sys.stdout = sys.__stdout__

    def save_to_txt_file(self):

        with open(self._path, self._write_mode) as out_file:
            out_file.write(self._result.getvalue())


def calculate_coint_results(merged_prices_comb_df: pd.DataFrame, ticker, min_period_yrs: float, max_half_life: int,
                            min_half_life: float, save_price_df: bool = True, save_all: bool = False,
                            print_verbose: bool = True, print_file: bool = True, alt_cols=None, time_interval='daily'):

    # compute the johansen test
    # important note: johansen test is asymptotic and only valid for large samples (>40),
    # research use of Phillips–Ouliaris cointegration test for smaller samples
    n_samples = merged_prices_comb_df.shape[0]

    # get the max and min date
    n_trading_days_per_year = 252
    n_trading_weeks_per_year = 52
    max_date = merged_prices_comb_df.index.max()
    min_date = merged_prices_comb_df.index.min()

    if time_interval == 'daily':
        n_sample_min = n_trading_days_per_year * min_period_yrs
    elif time_interval == 'weekly':
        n_sample_min = n_trading_weeks_per_year * min_period_yrs

    if n_samples < n_sample_min:

        if print_verbose:
            print('Insufficient samples detected')

        if save_all:
            valid_dict = create_dict(ticker=ticker, min_date=min_date, max_date=max_date, n_samples=n_samples,
                                     merged_prices_comb_df=merged_prices_comb_df, sample_pass=False,
                                     comment='Insufficient data samples', save_price_df=save_price_df)
            return valid_dict

        else:
            return None

    s_file_path = save_file_path('coint_results', '_'.join(ticker) + '.txt') if print_file else None
    std_o = StdoutRedirection(s_file_path)
    with std_o:
        try:
            result = coint_johansen(merged_prices_comb_df, 0, 2)
        except np.linalg.LinAlgError:
            if print_verbose:
                print('Singular matrix detected')
            if save_all:
                valid_dict = create_dict(ticker=ticker, min_date=min_date, max_date=max_date, n_samples=n_samples,
                                         merged_prices_comb_df=merged_prices_comb_df, sample_pass=False,
                                         comment='Johansen test singular matrix detected', save_price_df=save_price_df)
                return valid_dict
            else:
                return None

    # check trace and eigen statistics for r=0 (all we care about is the strongest relationship)
    johansen_90_p_trace = result.cvt[0, 0]
    johansen_95_p_trace = result.cvt[0, 1]
    johansen_99_p_trace = result.cvt[0, 2]
    johansen_trace_stat = result.lr1[0]

    johansen_90_p_eigen = result.cvm[0, 0]
    johansen_95_p_eigen = result.cvm[0, 1]
    johansen_99_p_eigen = result.cvm[0, 2]
    johansen_eigen_stat = result.lr2[0]

    johansen_eigenvectors = result.evec[:, 0]
    johansen_eigenvalue = result.eig[0]

    if alt_cols:
        cols = alt_cols
    else:
        cols = merged_prices_comb_df.columns.tolist()
    p_weights = pd.Series(johansen_eigenvectors, index=cols)

    # TODO: consider extracting trace and eigen stats for r<=1, r<=2, etc

    # check if trace and eigen stat values of johansen test are greater than 95% crit level
    if (johansen_trace_stat > johansen_95_p_trace) and (johansen_eigen_stat > johansen_95_p_eigen):

        # calculate ADF test and compute half life of portfolio
        merged_prices_comb_df['portfolio_price'] = merged_prices_comb_df.dot(p_weights)

        # cadf
        cadf = ts.adfuller(merged_prices_comb_df.portfolio_price)

        with std_o:
            print('\nADF Statistic: %f' % cadf[0])
            print('--------------------------------------------------')
            print('p-value: %f' % cadf[1])
            print('--------------------------------------------------')
            print('Critical Values:')
            print('--------------------------------------------------')
            for key, value in cadf[4].items():
                print('\t%s: %.3f' % (key, value))

        adf_test_stat = abs(cadf[0])
        adf_99_p_stat = abs(cadf[4]['1%'])
        adf_95_p_stat = abs(cadf[4]['5%'])
        adf_90_p_stat = abs(cadf[4]['10%'])

        hurst_exp = hurst(merged_prices_comb_df.portfolio_price)
        half_life_ = half_life(merged_prices_comb_df.portfolio_price)

        if adf_test_stat <= adf_95_p_stat:
            sample_pass = False
            comment = 'Fails the ADF tests for co-integration'
            if print_verbose:
                print('ADF null hypotheses not rejected')
        elif hurst_exp >= 0.5:
            sample_pass = False
            comment = 'Hurst exp is greater than 0.5 - time trend exhibited'
            if print_verbose:
                print('Hurst exp is greater than 0.5')
        elif half_life_ > max_half_life:
            sample_pass = False
            comment = 'Half life greater than max specified'
            if print_verbose:
                print('Half life greater than max specified')
        elif half_life_ < min_half_life:
            sample_pass = False
            comment = 'Half life smaller than min specified'
            if print_verbose:
                print('Half life smaller than min specified')
        else:
            sample_pass = True
            comment = None
            if print_verbose:
                print('Validation sample passes stationary checks')

            # plot the price portfolio
            if print_file:
                sns_plot = sns.lineplot(data=merged_prices_comb_df.portfolio_price)
                sns_plot.figure.savefig(save_file_path('plots', '_'.join(ticker) + '.png'), format='png', dpi=1000)
                sns_plot.figure.clf()

        if save_all or sample_pass:
            valid_dict = create_dict(ticker=ticker, min_date=min_date, max_date=max_date, n_samples=n_samples,
                                     merged_prices_comb_df=merged_prices_comb_df, sample_pass=sample_pass,
                                     johansen_90_p_trace=johansen_90_p_trace, johansen_95_p_trace=johansen_95_p_trace,
                                     johansen_99_p_trace=johansen_99_p_trace, johansen_trace_stat=johansen_trace_stat,
                                     johansen_90_p_eigen=johansen_90_p_eigen, johansen_95_p_eigen=johansen_95_p_eigen,
                                     johansen_99_p_eigen=johansen_99_p_eigen, johansen_eigen_stat=johansen_eigen_stat,
                                     johansen_eigenvectors=p_weights,
                                     johansen_eigenvalue=johansen_eigenvalue, adf_test_stat=adf_test_stat,
                                     adf_99_p_stat=adf_99_p_stat, adf_95_p_stat=adf_95_p_stat,
                                     adf_90_p_stat=adf_90_p_stat,
                                     hurst_exp=hurst_exp, half_life_=half_life_, comment=comment, save_price_df=save_price_df)

            if print_file:
                std_o.save_to_txt_file()
            return valid_dict
        else:
            return None

    else:
        if save_all:
            valid_dict = create_dict(ticker=ticker, min_date=min_date, max_date=max_date, n_samples=n_samples,
                                     merged_prices_comb_df=merged_prices_comb_df, sample_pass=False,
                                     johansen_90_p_trace=johansen_90_p_trace, johansen_95_p_trace=johansen_95_p_trace,
                                     johansen_99_p_trace=johansen_99_p_trace, johansen_trace_stat=johansen_trace_stat,
                                     johansen_90_p_eigen=johansen_90_p_eigen, johansen_95_p_eigen=johansen_95_p_eigen,
                                     johansen_99_p_eigen=johansen_99_p_eigen, johansen_eigen_stat=johansen_eigen_stat,
                                     johansen_eigenvectors=p_weights, johansen_eigenvalue=johansen_eigenvalue,
                                     comment='Fails the Johansen tests for co-integration', save_price_df=save_price_df)
            if print_file:
                std_o.save_to_txt_file()
            return valid_dict
        else:
            return None



########################################################

#####################Run main code######################


schema_output = StructType([
        StructField('ticker', ArrayType(StringType(), True), True),
        StructField('max_date', DateType(), True),
        StructField('min_date', DateType(), True),
        StructField('n_samples', IntegerType(), True),
        StructField('johansen_90_p_trace', FloatType(), True),
        StructField('johansen_95_p_trace', FloatType(), True),
        StructField('johansen_99_p_trace', FloatType(), True),
        StructField('johansen_trace_stat', FloatType(), True),
        StructField('johansen_90_p_eigen', FloatType(), True),
        StructField('johansen_95_p_eigen', FloatType(), True),
        StructField('johansen_99_p_eigen', FloatType(), True),
        StructField('johansen_eigen_stat', FloatType(), True),
        StructField('johansen_eigenvectors', ArrayType(FloatType(), True), True),
        StructField('johansen_eigenvalue', FloatType(), True),
        StructField('adf_test_stat', FloatType(), True),
        StructField('adf_99_p_stat', FloatType(), True),
        StructField('adf_95_p_stat', FloatType(), True),
        StructField('adf_90_p_stat', FloatType(), True),
        StructField('hurst_exp', FloatType(), True),
        StructField('half_life_', FloatType(), True),
        StructField('sample_pass', IntegerType(), True),
        StructField('comment', StringType(), True)
    ])

@pandas_udf(schema_output, PandasUDFType.GROUPED_MAP)
def process_pairs_spark(df):

    ticker = df['combination'].iloc[0].split("_")
    ticker_col = [col for col in df if col.startswith('x')]

    merged_prices_comb_df = df.loc[:, df.columns != 'combination']
    merged_prices_comb_df.set_index('formatted_date', inplace=True)

    result_dict = calculate_coint_results(merged_prices_comb_df=merged_prices_comb_df,
                                          ticker=ticker,
                                          min_period_yrs=min_period_yrs,
                                          max_half_life=max_half_life,
                                          min_half_life=min_half_life,
                                          save_price_df=False,
                                          save_all=True,
                                          print_verbose=False,
                                          print_file=False,
                                          alt_cols=ticker_col,
                                          time_interval=time_interval)

    # return results dataframe
    results_df = pd.DataFrame()
    results_df = results_df.append(result_dict, ignore_index=True)
    return results_df


t_0 = time()

df_map = df.groupby("combination").apply(process_pairs_spark)
# df_map.show()
results_df_p = df_map.toPandas()
results_df_p.to_csv(save_path, mode='w', index=True)

print(np.round(time() - t_0, 3), "seconds elapsed...")