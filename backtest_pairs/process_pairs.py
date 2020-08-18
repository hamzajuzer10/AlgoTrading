import pandas as pd
from yahoofinancials import YahooFinancials
import itertools
from itertools import combinations
import os
import sys
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import seaborn as sns
from backtest_pairs.johansen import coint_johansen
from backtest_pairs.hurst_half_life import hurst, half_life
from backtest_pairs.utils import save_file_path, create_dict
import warnings
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

etf_ticker_path = 'C:\\Users\\hamzajuzer\\Documents\\Algorithmic Trading\\AlgoTradingv1\\backtest_pairs\\etf_tickers.csv'
num_tickers_in_basket = 3 # max 12
start_date = '2006-01-01'
end_date = '2012-12-31'
time_interval = 'daily'
min_period_yrs = 3
max_half_life = 30 # in time interval units
min_half_life = 0 # in time interval units


class StdoutRedirection:
    """Standard output redirection context manager"""

    def __init__(self, path, write_mode="w"):
        self._path = path
        self._write_mode = write_mode

    def __enter__(self):
        sys.stdout = open(self._path, mode=self._write_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = sys.__stdout__


def import_ticker_data(ticker_file_path:str, start_date:str, end_date:str, time_interval: str):

    # read csv file with ticker data
    ticker_df = pd.read_csv(ticker_file_path)
    ticker_list = ticker_df['Symbol'].tolist()

    yahoo_financials = YahooFinancials(ticker_list)

    # download data
    data = yahoo_financials.get_historical_price_data(start_date=start_date,
                                                      end_date=end_date,
                                                      time_interval=time_interval)

    # get the prices as a dataframe
    for ticker in data.keys():

        # if prices is a key
        try:
            price_df = pd.DataFrame(data[ticker]['prices'])
            price_df = price_df.drop('date', axis=1).set_index('formatted_date')
            price_df = price_df[['adjclose']]
            price_df.rename(columns={'adjclose': ticker}, inplace=True)
            data[ticker]['price_df'] = price_df
        except KeyError:
            warnings.warn('Price data not available for {}!'.format(ticker))
            warnings.warn('Removing {} from data list'.format(ticker))
            del data[ticker]

    # TODO: research if Xccy needs to be the same for all ticker prices - currently keeping Xccy in base units

    return data


def create_valid_ticker_combs(ticker_data, min_period_yrs: float, num_tickers_in_basket: int,
                              max_half_life: int, min_half_life: float):

    # create num_tickers_in_basket combinations of ticker data
    comb = combinations(ticker_data.keys(), num_tickers_in_basket)
    list_comb = list(comb)

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


        # compute the johansen test
        # important note: johansen test is asymptotic and only valid for large samples (>40),
        # research use of  Phillips–Ouliaris cointegration test for smaller samples
        merged_prices_comb_df.dropna(inplace=True)
        n_samples = merged_prices_comb_df.shape[0]

        # get the max and min date
        n_trading_days_per_year = 252
        max_date = merged_prices_comb_df.index.max()
        min_date = merged_prices_comb_df.index.min()

        if n_samples < (n_trading_days_per_year*min_period_yrs):
            valid_dict = create_dict(ticker=i, min_date=min_date, max_date=max_date, n_samples=n_samples,
                                     merged_prices_comb_df=merged_prices_comb_df, sample_pass=False,
                                     comment='Insufficient data samples')
            valid_combinations = valid_combinations.append(valid_dict, ignore_index=True)
            print('Insufficient samples detected for combination {a}/{b}: {c}'.format(a=index, b=len(list_comb), c=i))
            continue

        with StdoutRedirection(save_file_path('results', '_'.join(i)+'.txt')):
            result = coint_johansen(merged_prices_comb_df, 0, 2)

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

        cols = merged_prices_comb_df.columns.tolist()
        p_weights = pd.Series(johansen_eigenvectors, index=cols)

        # TODO: consider extracting trace and eigen stats for r<=1, r<=2, etc

        # check if trace and eigen stat values of johansen test are greater than 95% crit level
        if (johansen_trace_stat > johansen_95_p_trace) and (johansen_eigen_stat > johansen_95_p_eigen):

            # calculate ADF test and compute half life of portfolio
            merged_prices_comb_df['portfolio_price'] = merged_prices_comb_df.dot(p_weights)

            # cadf
            cadf = ts.adfuller(merged_prices_comb_df.portfolio_price)

            with StdoutRedirection(save_file_path('results', '_'.join(i) + '.txt'), write_mode="a"):
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

            if adf_test_stat <=  adf_95_p_stat:
                sample_pass = False
                comment = 'Fails the ADF tests for co-integration'
                print('ADF null hypotheses not rejected for combination {a}/{b}: {c}'.format(a=index, b=len(list_comb), c=i))
            elif hurst_exp >= 0.5:
                sample_pass = False
                comment = 'Hurst exp is greater than 0.5 - time trend exhibited'
                print('Hurst exp is greater than 0.5 for combination {a}/{b}: {c}'.format(a=index, b=len(list_comb), c=i))
            elif half_life_ > max_half_life:
                sample_pass = False
                comment = 'Half life greater than max specified'
                print('Half life greater than max specified for combination {a}/{b}: {c}'.format(a=index, b=len(list_comb), c=i))
            elif half_life_ < min_half_life:
                sample_pass = False
                comment = 'Half life smaller than min specified'
                print('Half life smaller than min specified for combination {a}/{b}: {c}'.format(a=index, b=len(list_comb), c=i))
            else:
                sample_pass = True
                comment = None

                # plot the price portfolio
                sns_plot = sns.lineplot(data=merged_prices_comb_df.portfolio_price)
                sns_plot.figure.savefig(save_file_path('plots', '_'.join(i) + '.png'), format='png', dpi=1000)
                sns_plot.figure.clf()

            valid_dict = create_dict(ticker=i, min_date=min_date, max_date=max_date, n_samples=n_samples,
                                     merged_prices_comb_df=merged_prices_comb_df, sample_pass=sample_pass,
                                     johansen_90_p_trace=johansen_90_p_trace, johansen_95_p_trace=johansen_95_p_trace,
                                     johansen_99_p_trace=johansen_99_p_trace, johansen_trace_stat=johansen_trace_stat,
                                     johansen_90_p_eigen=johansen_90_p_eigen, johansen_95_p_eigen=johansen_95_p_eigen,
                                     johansen_99_p_eigen=johansen_99_p_eigen, johansen_eigen_stat=johansen_eigen_stat,
                                     johansen_eigenvectors=p_weights,
                                     johansen_eigenvalue=johansen_eigenvalue, adf_test_stat=adf_test_stat,
                                     adf_99_p_stat=adf_99_p_stat, adf_95_p_stat=adf_95_p_stat, adf_90_p_stat=adf_90_p_stat,
                                     hurst_exp=hurst_exp, half_life_=half_life_, comment=comment)
            valid_combinations = valid_combinations.append(valid_dict, ignore_index=True)
            print('Validation sample passes stationary checks for combination {a}/{b}: {c}'.format(a=index, b=len(list_comb),
                                                                                             c=i))

        else:

            valid_dict = create_dict(ticker=i, min_date=min_date, max_date=max_date, n_samples=n_samples,
                                     merged_prices_comb_df=merged_prices_comb_df, sample_pass=False,
                                     johansen_90_p_trace=johansen_90_p_trace, johansen_95_p_trace=johansen_95_p_trace,
                                     johansen_99_p_trace=johansen_99_p_trace, johansen_trace_stat=johansen_trace_stat,
                                     johansen_90_p_eigen=johansen_90_p_eigen, johansen_95_p_eigen=johansen_95_p_eigen,
                                     johansen_99_p_eigen=johansen_99_p_eigen, johansen_eigen_stat=johansen_eigen_stat,
                                     johansen_eigenvectors=p_weights, johansen_eigenvalue=johansen_eigenvalue,
                                     comment='Fails the Johansen tests for co-integration')
            valid_combinations = valid_combinations.append(valid_dict, ignore_index=True)
            print('Johansen null hypothesis not rejected for combination {a}/{b}: {c}'.format(a=index, b=len(list_comb), c=i))

    return valid_combinations


if __name__== '__main__':

    # download and import data
    ticker_data = import_ticker_data(etf_ticker_path,
                                     start_date=start_date,
                                     end_date=end_date,
                                     time_interval=time_interval)

    valid_combinations = create_valid_ticker_combs(ticker_data, min_period_yrs=min_period_yrs,
                                                   num_tickers_in_basket=num_tickers_in_basket,
                                                   max_half_life=max_half_life, min_half_life=min_half_life)

    valid_combinations.to_pickle(save_file_path('results', 'results_df.pkl'))