import pandas as pd
from backtest_pairs.yfinance_connector import import_ticker_data, load_ticker_data_json
from itertools import combinations
from io import StringIO
import sys
import statsmodels.tsa.stattools as ts
import seaborn as sns
from backtest_pairs.johansen import coint_johansen
from backtest_pairs.hurst_half_life import hurst, half_life
from backtest_pairs.utils import save_file_path, create_dict
import warnings
from collections import defaultdict
import numpy as np
import os

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

json_path = "/backtest_pairs/data/etf_tickers_07_2020.json"
num_tickers_in_basket = 2 # max 12
start_date = '2006-04-26'
end_date = '2012-04-09'
time_interval = 'daily'
min_period_yrs = 1.5
max_half_life = 30 # in time interval units
min_half_life = 2 # in time interval units
time_zones = [-14400]
use_close_prices = False


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


def filter_high_liquidity_tickers(data: dict, n_days = 90, min_last_n_day_vol: float = 10000):

    # filter tickers with average 30 days vol > min
    for ticker in data:
        try:
            vol_df = pd.DataFrame(data[ticker]['prices'])
            vol_df = vol_df.drop('date', axis=1).set_index('formatted_date')
            vol_df.index = pd.to_datetime(vol_df.index, format='%Y-%m-%d %H:%M:%S')
            vol_df = vol_df[['volume']]
            vol_mean = np.mean(vol_df.tail(n_days)).item()
            if vol_mean < min_last_n_day_vol:
                print('Ticker {b} volume traded is smaller than min, {a}. Disregarding ticker!'.format(b=ticker, a=min_last_n_day_vol))
                del data[ticker]

        except KeyError:
            warnings.warn('Volume data not available for {}!'.format(ticker))
            del data[ticker]

        return data


def build_price_df(data: dict, use_close_prices: bool = False, time_interval = 'daily'):

    # yahoo adjusts close prices for stock splits
    # and adjusted close prices for stock splits and dividends

    output_dict = {}

    # if prices, currency, instrumentType and timeZone is a key
    for ticker in data:
        try:
            price_df = pd.DataFrame(data[ticker]['prices'])
            price_df = price_df.drop('date', axis=1).set_index('formatted_date')
            price_df.index = pd.to_datetime(price_df.index, format='%Y-%m-%d %H:%M:%S')

            # exclude dividends from prices if set to true
            if use_close_prices:
                price_df = price_df[['close']]
                price_df.rename(columns={'close': ticker}, inplace=True)
            else:
                price_df = price_df[['adjclose']]
                price_df.rename(columns={'adjclose': ticker}, inplace=True)

            if time_interval == 'weekly':

                # resample at weekly time intervals (keep monday only)
                price_df = price_df[price_df.index.weekday == 0]

            output_dict[ticker] = {}
            output_dict[ticker]['price_df'] = price_df
            output_dict[ticker]['currency'] = data[ticker]['currency']
            output_dict[ticker]['instrumentType'] = data[ticker]['instrumentType']
            output_dict[ticker]['timeZone_gmtOffset'] = data[ticker]['timeZone']['gmtOffset']

        except KeyError:
            warnings.warn('Price, currency, instrumentType or timeZone data not available for {}!'.format(ticker))

    return output_dict


def group_timeZone(data: dict):

    # group tickers with the same timezone
    output = defaultdict(list)

    for ticker, value in data.items():

        output[value['timeZone_gmtOffset']].append(ticker)

    return output


def create_combinations(data: dict, n_tickers_in_basket: int):

    # create combinations
    valid_combinations = []
    for key, value in data.items():
        comb = combinations(value, n_tickers_in_basket)
        valid_combinations.extend(list(comb))

    return valid_combinations


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


def create_valid_ticker_combs(ticker_data, min_period_yrs: float, num_tickers_in_basket: int,
                              max_half_life: int, min_half_life: float, time_zones=None, save_all=True, time_interval='daily',
                              use_close_prices: bool = False, min_liq_n_days=90, min_liq_last_n_day_vol=10000,
                              force_through=False):

    if not force_through:
        # only consider tickers with sufficient liquidity
        ticker_data = filter_high_liquidity_tickers(ticker_data, n_days=min_liq_n_days,
                                                    min_last_n_day_vol=min_liq_last_n_day_vol)

    # create a price df, timezone and currency for each ticker
    ticker_data = build_price_df(ticker_data, time_interval=time_interval, use_close_prices=use_close_prices)

    # create num_tickers_in_basket combinations of ticker data, grouping by timeZone
    time_zone_ticker_groups = group_timeZone(ticker_data)

    # Filter on timezones for faster processing
    if time_zones:
        time_zone_ticker_groups = dict((k, time_zone_ticker_groups[k]) for k in time_zones if k in time_zone_ticker_groups)

    list_comb = create_combinations(time_zone_ticker_groups, num_tickers_in_basket)

    if len(list_comb) == 0:
        warnings.warn('No combination of tickers generated!')

    # For each combination, check the minimum start dates for all tickers in basket
    valid_combinations = pd.DataFrame()
    for index, i in enumerate(list_comb):
        # johansen test
        # merge all dataframes together
        print('Processing combination {a}/{b}: {c}'.format(a=index, b=len(list_comb), c=i))
        merged_prices_comb_df = pd.DataFrame()
        for j in i:

            merged_prices_comb_df = pd.merge(merged_prices_comb_df, ticker_data[j]['price_df'], left_index=True,
                                             right_index=True, how='outer')

        merged_prices_comb_df.dropna(inplace=True)

        if not force_through:
            result_dict = calculate_coint_results(merged_prices_comb_df=merged_prices_comb_df,
                                                  ticker=i,
                                                  min_period_yrs=min_period_yrs,
                                                  max_half_life=max_half_life,
                                                  min_half_life=min_half_life,
                                                  save_price_df=True,
                                                  save_all=save_all,
                                                  print_verbose=True,
                                                  print_file=True,
                                                  time_interval=time_interval)
        else:
            result_dict = create_dict(ticker=i, min_date=None, max_date=None, n_samples=None,
                                      merged_prices_comb_df=merged_prices_comb_df, sample_pass=True,
                                      johansen_90_p_trace=None, johansen_95_p_trace=None,
                                      johansen_99_p_trace=None, johansen_trace_stat=None,
                                      johansen_90_p_eigen=None, johansen_95_p_eigen=None,
                                      johansen_99_p_eigen=None, johansen_eigen_stat=None,
                                      johansen_eigenvectors=None,
                                      johansen_eigenvalue=None, adf_test_stat=None,
                                      adf_99_p_stat=None, adf_95_p_stat=None,
                                      adf_90_p_stat=None,
                                      hurst_exp=None, half_life_=None, comment="Force through ticker", save_price_df=True)

        if result_dict:
            valid_combinations = valid_combinations.append(result_dict, ignore_index=True)

    return valid_combinations


if __name__== '__main__':

    # download and import data
    print('Importing ticker data')
    # ticker_data = import_ticker_data(tickers=['EWA', 'EWC', 'DIA', 'IYT'],
    #                                  start_date=start_date,
    #                                  end_date=end_date,
    #                                  time_interval='daily')

    ticker_data = load_ticker_data_json(json_path)

    # calculating valid ticker combinations
    print('Calculating valid ticker combinations')
    valid_combinations = create_valid_ticker_combs(ticker_data, min_period_yrs=min_period_yrs,
                                                   num_tickers_in_basket=num_tickers_in_basket,
                                                   max_half_life=max_half_life, min_half_life=min_half_life,
                                                   time_zones=time_zones, save_all=False, time_interval=time_interval,
                                                   use_close_prices=use_close_prices)

    # saving valid ticker combinations
    print('Saving results')
    valid_combinations.to_pickle(save_file_path('coint_results', 'valid_coint_results_df.pkl'))