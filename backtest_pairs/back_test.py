import backtrader as bt
from backtest_pairs.back_test_strategy import PyKalman_PairTradingStrategy, get_inv_perc_diff_df
from backtest_pairs.back_test_sizer import exampleSizer, printSizingParams, maxRiskSizer
import pandas as pd
from backtest_pairs.process_pairs import import_ticker_data, create_valid_ticker_combs, build_price_df
from backtest_pairs.kalman_filter import reformat_data
from backtest_pairs.utils import save_file_path
import sys
import warnings
import numpy as np
import quantstats as qs
import os
import csv
from datetime import datetime

etf_ticker_path = '/backtest_pairs/data/etf_tickers_07_2020.csv'
start_date = '2020-06-01'
end_date = '2021-05-06'
time_interval = 'weekly'
time_zones = [-14400, 0]
num_tickers_in_basket = 2
min_period_yrs = 1.5
max_half_life = 12 # in time interval units, 30 if days, 12 if weeks
min_half_life = 2 # in time interval units, 2 is default
use_close_prices = True

class CashMarket(bt.analyzers.Analyzer):

    def start(self):
        super(CashMarket, self).start()

    def create_analysis(self):
        self.rets = {}
        self.vals = 0.0

    def notify_cashvalue(self, cash, value):
        self.vals = (cash, value)
        self.rets[self.strategy.datetime.datetime().strftime("%Y-%m-%d")] = self.vals

    def get_analysis(self):
        return self.rets


class PandasData(bt.feed.DataBase):
    '''
    The ``dataname`` parameter inherited from ``feed.DataBase`` is the pandas
    DataFrame
    '''

    params = (
        # Possible values for datetime (must always be present)
        #  None : datetime is the "index" in the Pandas Dataframe
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the column in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('datetime', None),

        # Possible values below:
        #  None : column not present
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the column in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('open', None),
        ('high', None),
        ('low', None),
        ('close', -1),
        ('volume', None),
        ('openinterest', None),
    )


class CommIB(bt.CommInfoBase): # Interactive broker tiered US commission charges
    params = (
        ('commission', 0.0035),
        ('stocklike', True),  # Stocks
        ('commtype', bt.CommInfoBase.COMM_FIXED),  # Apply fixed Commission
        ('min_comm', 0.35),  # USD
        ('max_comm', 0.005),
        ('on_slippage', 0.001),
    )

    def _getcommission(self, size, price, pseudoexec):

        commis = abs(size) * self.p.commission
        trade_value = abs(size) * price
        max_commis = trade_value * self.p.max_comm
        slipp_value = trade_value * self.p.on_slippage

        if self._commtype == self.COMM_PERC:
            return trade_value * self.p.commission

        if commis < self.p.min_comm:
            return self.p.min_comm + trade_value * self.p.on_slippage
        elif commis > max_commis:
            return max_commis + slipp_value
        else:
            return commis + slipp_value


def runstrategy(valid_combinations: pd.Series
                , benchmark_data: pd.Series
                , cash=10000
                , results_folder_name='results'
                , min_final_pfolio_value_for_tearsheet_perc=1.2
                , min_Sharpe_for_tearsheet=2.5
                , min_date: str = None
                , max_date: str = None
                , short_interest=0.05
                , time_interval='daily'
                , use_benchmark_prices=False):

    print('Running strategy on {} combination'.format(valid_combinations['ticker']))

    # Create a cerebro
    cerebro = bt.Cerebro()

    # set cheat on close (to execute at current close)
    cerebro.broker.set_coc(True)

    # Create data feeds - y variable
    # rename to close
    y_data = valid_combinations['merged_prices_comb_df'][[valid_combinations['y_ticker']]]

    if min_date:

        # cap y at min date
        y_data = y_data.loc[min_date:]

    if max_date:

        # cap y at max date
        y_data = y_data.loc[:max_date]

    y_data.rename(columns={y_data.columns[0]: "close"}, inplace=True)
    data = bt.feeds.PandasData(dataname=y_data,
                               name=valid_combinations['y_ticker'])

    # Add the data to cerebro
    cerebro.adddata(data)

    # Create data feeds - x variables
    for ticker in valid_combinations['x_ticker']:
        x_data = valid_combinations['merged_prices_comb_df'][[ticker]]

        if min_date:
            # cap x at min date
            x_data = x_data.loc[min_date:]

        if max_date:
            # cap x at max date
            x_data = x_data.loc[:max_date]


        x_data.rename(columns={x_data.columns[0]: "close"}, inplace=True)
        data = bt.feeds.PandasData(dataname=x_data,
                                   name=ticker)

        # Add the data to cerebro
        cerebro.adddata(data)

    # Add the strategy
    cerebro.addstrategy(PyKalman_PairTradingStrategy,
                        y_ticker=valid_combinations['y_ticker'],
                        x_ticker=valid_combinations['x_ticker'],
                        state_mean=np.asarray(valid_combinations['initial_mean']),
                        delta=0.0001,
                        n_dim_state=valid_combinations['dim'],
                        Ve=0.001,
                        kalman_averaging=False,
                        initialisation_period=30)

    # Add the cash
    cerebro.broker.setcash(cash)

    # Add the commission - and add interest for short positions
    comminfo = CommIB(interest=short_interest)

    cerebro.broker.addcommissioninfo(comminfo)

    # add analyzer
    cerebro.addanalyzer(CashMarket, _name='cashmarket')

    # Print out the starting conditions
    starting_pfolio_value = cerebro.broker.getvalue()
    print('Starting Portfolio Value: %.2f' % starting_pfolio_value)

    # And run it
    results = cerebro.run()

    # Print out the final result
    final_pfolio_value = cerebro.broker.getvalue()
    print('Final Portfolio Value: %.2f' % final_pfolio_value)

    # ---- Format the values from results ----
    df_values = pd.DataFrame(results[0].analyzers.getbyname("cashmarket").get_analysis()).T
    df_values = df_values.iloc[:, 1]
    returns = qs.utils.to_returns(df_values)
    returns.index = pd.to_datetime(returns.index)

    # start from the first non-zero return entry in the series - removes all returns during initialisation period
    start = returns.loc[returns != 0].index
    returns = returns[returns.index >= start.min()]

    if time_interval == 'daily':
        n_periods = 252
    elif time_interval == 'weekly':
        n_periods = 52

    mean_returns = returns.mean()
    std_returns = returns.std()

    neg_returns = returns[returns < 0]
    neg_std_returns = neg_returns.std()

    # Calculate monthly Sharpe and Sortino ratios
    if (std_returns != 0) and std_returns:
        Sharpe = (mean_returns/std_returns)*np.sqrt(n_periods)

    else:
        Sharpe = None

    if (neg_std_returns != 0) and neg_std_returns:
        Sortino = (mean_returns/neg_std_returns)*np.sqrt(n_periods)
    else:
        Sortino = None

    print("Sharpe ratio: {}".format(Sharpe))
    print("Sortino ratio: {}".format(Sortino))

    # save the result only if the final pfolio value is greater than starting_pfolio_value * min_final_pfolio_value_for_tearsheet_perc
    if (final_pfolio_value >= starting_pfolio_value*min_final_pfolio_value_for_tearsheet_perc) and \
            (Sharpe >= min_Sharpe_for_tearsheet):

        # ----------------------------------------

        # ---- Format the benchmark from SPY.csv ----
        if min_date:
            # cap at min date
            benchmark_data = benchmark_data.loc[min_date:]

        if max_date:
            # cap at max date
            benchmark_data = benchmark_data.loc[:max_date]

        # -------------------------------------------

        qs.extend_pandas()
        output_file_name = "qs_" + 'y_' + valid_combinations['y_ticker'] + '_x_' + '_'.join(valid_combinations['x_ticker']) + '.html'
        output_file_name = save_file_path(folder_name=results_folder_name, filename=output_file_name, wd=None)

        if use_benchmark_prices:
            benchmark_data = get_inv_perc_diff_df(benchmark_data, init_date=start.min(), time_interval=time_interval)

        try:
            qs.reports.html(returns, benchmark=benchmark_data, output=output_file_name)

        except:
            print('Error generating report for {} combination'.format(valid_combinations['ticker']))
            print('Unexpected error:{}'.format(sys.exc_info()[0]))

    return starting_pfolio_value, final_pfolio_value, \
           mean_returns, std_returns, neg_std_returns, \
           Sharpe, Sortino


if __name__ == '__main__':

    # download and import data
    print('Importing ticker data')
    ticker_data = import_ticker_data(tickers=['EIS', 'PLTM'],
                                     start_date=start_date,
                                     end_date=end_date,
                                     time_interval='daily')

    # benchmark data
    benchmark_ticker = 'SPY' #SPY, ^VIX
    print('Importing benchmark data - {}'.format(benchmark_ticker))
    benchmark_data = import_ticker_data(tickers=[benchmark_ticker],
                                        start_date=start_date,
                                        end_date=end_date,
                                        time_interval='daily')

    benchmark_data = build_price_df(benchmark_data, time_interval=time_interval, use_close_prices=use_close_prices)

    # calculating valid ticker combinations
    print('Calculating valid ticker combinations')
    valid_combinations = create_valid_ticker_combs(ticker_data, min_period_yrs=min_period_yrs,
                                                   num_tickers_in_basket=num_tickers_in_basket,
                                                   max_half_life=max_half_life, min_half_life=min_half_life,
                                                   time_zones=time_zones, save_all=True, time_interval=time_interval,
                                                   use_close_prices=use_close_prices, min_liq_n_days=180,
                                                   min_liq_last_n_day_vol=1000, force_through=True)

    if valid_combinations.shape[0] == 0:
        warnings.warn('No valid ticker combinations to process!')
        sys.exit(0)

    # Filter only on valid combinations
    print('Filtering valid ticker combinations only')
    valid_combinations = valid_combinations.loc[valid_combinations['sample_pass'] == True]

    if valid_combinations.shape[0] == 0:
        warnings.warn('No valid ticker combinations to process!')
        sys.exit(0)

    # For each valid combination
    print('Reformatting valid ticker combination data')
    valid_combinations = reformat_data(valid_combinations, zero_mean=True)

    # # Load valid combination from file
    # print('Loading valid ticker combinations')
    # valid_combinations = pd.read_pickle("C:\\Users\\hamzajuzer\\Documents\\Algorithmic Trading\\AlgoTradingv1\\backtest_pairs\\coint_results\\kalman_results_df_12_2020_weekly.pkl")

    if valid_combinations.shape[0] == 0:
        warnings.warn('No valid ticker combinations to process!')
        sys.exit(0)

    # # Filter only combinations where estimated Sharpe Ratio >2 and APRs> 25%
    # valid_combinations = valid_combinations[valid_combinations.Sharpe >= 2]
    # valid_combinations = valid_combinations[valid_combinations.APR >= 0.25]

    print('Running strategy on valid ticker combination data')
    valid_combinations['starting_pfolio_value_backtest'], \
    valid_combinations['final_pfolio_value_backtest'], \
    valid_combinations['mean_returns'], \
    valid_combinations['std_returns'], \
    valid_combinations['neg_std_returns'], \
    valid_combinations['Sharpe'], \
    valid_combinations['Sortino'] = zip(*valid_combinations.apply(runstrategy,
                                                  args=(benchmark_data[benchmark_ticker]['price_df'], 10000, 'results_ind_12_2020_weekly', 0, 0, None, end_date, 0.05, time_interval, False), axis=1))

    # # save valid combinations
    # print('Saving results')
    # valid_combinations.to_pickle(
    #     save_file_path(folder_name='results_12_2020_weekly', filename='backtest_results_df.pkl'))
