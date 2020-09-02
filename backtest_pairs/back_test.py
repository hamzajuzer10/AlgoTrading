import backtrader as bt
from backtest_pairs.back_test_strategy import PyKalman_PairTradingStrategy
from backtest_pairs.back_test_sizer import exampleSizer, printSizingParams, maxRiskSizer
import pandas as pd
from backtest_pairs.process_pairs import import_ticker_data, create_valid_ticker_combs
from backtest_pairs.kalman_filter import reformat_data
import sys
import warnings
import numpy as np
import quantstats as qs
import os
import csv
from datetime import datetime


etf_ticker_path = '/backtest_pairs/data/etf_tickers.csv'
start_date = '2006-04-26'
end_date = '2012-04-09'
time_interval = 'daily'
num_tickers_in_basket = 2
min_period_yrs = 3
max_half_life = 30 # in time interval units
min_half_life = 0 # in time interval units


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
                , short_interest=0.0125):

    print('Running strategy on {} combination'.format(valid_combinations['ticker']))

    # Create a cerebro
    cerebro = bt.Cerebro()

    # set cheat on close (to execute at current close)
    cerebro.broker.set_coc(True)

    # Create data feeds - y variable
    # rename to close
    y_data = valid_combinations['merged_prices_comb_df'][[valid_combinations['y_ticker']]]
    y_data.rename(columns={y_data.columns[0]: "close"}, inplace=True)
    data = bt.feeds.PandasData(dataname=y_data,
                               name=valid_combinations['y_ticker'])

    # Add the data to cerebro
    cerebro.adddata(data)

    # Create data feeds - x variables
    for ticker in valid_combinations['x_ticker']:
        x_data = valid_combinations['merged_prices_comb_df'][[ticker]]
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
                        state_covariance=np.zeros((valid_combinations['dim'], valid_combinations['dim'])),
                        observation_covariance=0.05,
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
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # And run it
    results = cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # ---- Format the values from results ----
    df_values = pd.DataFrame(results[0].analyzers.getbyname("cashmarket").get_analysis()).T
    df_values = df_values.iloc[:, 1]
    returns = qs.utils.to_returns(df_values)
    returns.index = pd.to_datetime(returns.index)
    # ----------------------------------------

    # ---- Format the benchmark from SPY.csv ----
    returns_bm = qs.utils.to_returns(benchmark_data)
    returns_bm.index = pd.to_datetime(returns_bm.index)
    # -------------------------------------------

    qs.extend_pandas()
    output_file_name = "qs_" + 'y_' + valid_combinations['y_ticker'] + '_x_' + '_'.join(valid_combinations['x_ticker']) + '.html'
    output_file_name = os.path.join("C:\\Users\\hamzajuzer\\Documents\\Algorithmic Trading\\AlgoTradingv1\\backtest_pairs\\results", output_file_name)
    qs.reports.html(returns, benchmark=benchmark_data, output=output_file_name)


if __name__ == '__main__':

    # download and import data
    print('Importing ticker data')
    ticker_data = import_ticker_data(ticker_file_path=etf_ticker_path,
                                     start_date=start_date,
                                     end_date=end_date,
                                     time_interval=time_interval)

    # benchmark data
    print('Importing benchmark data - SPY')
    benchmark_data = import_ticker_data(tickers=['SPY'],
                                        start_date=start_date,
                                        end_date=end_date,
                                        time_interval=time_interval)

    # Calculate portfolio stationarity to find valid combinations
    print('Calculating valid ticker combinations')
    valid_combinations = create_valid_ticker_combs(ticker_data, min_period_yrs=min_period_yrs,
                                                   num_tickers_in_basket=num_tickers_in_basket,
                                                   max_half_life=max_half_life, min_half_life=min_half_life)

    # Filter only on valid combinations
    print('Filtering valid ticker combinations only')
    valid_combinations = valid_combinations.loc[valid_combinations['sample_pass'] == True]

    if valid_combinations.shape[0] == 0:
        warnings.warn('No valid ticker combinations to process!')
        sys.exit(0)

    # For each valid combination
    print('Reformatting valid ticker combination data')
    valid_combinations['y_ticker'], valid_combinations['x_ticker'], \
    valid_combinations['initial_mean'], valid_combinations['dim'] = zip(
        *valid_combinations.apply(reformat_data, axis=1))

    # For each valid combination
    print('Running strategy on valid ticker combination data')
    valid_combinations = valid_combinations.apply(runstrategy,
                                                  args=(benchmark_data['SPY']['price_df'], 10000,), axis=1)
