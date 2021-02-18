import backtrader as bt
import numpy as np
from backtest_pairs.kalman_filter import MyKalmanPython
from math import sqrt, floor
import pandas as pd


def get_inv_perc_diff_df(df: pd.DataFrame, init_date, time_interval='daily'):


    # get the init value
    if time_interval == "daily":
        min_date = init_date + pd.Timedelta(days=1)
    elif time_interval == "weekly":
        min_date = init_date + pd.Timedelta(days=7)


    df = df[df.index >= (min_date)]

    # get min date in dataframe
    min_date = df.index.min()

    col_name = df.columns.tolist()[0]
    init_value = df[col_name].iloc[0]

    if time_interval == "daily":
        min_date = min_date - pd.Timedelta(days=1)
    elif time_interval == "weekly":
        min_date = min_date - pd.Timedelta(days=7)

    line = pd.DataFrame({col_name: init_value}, index=[min_date])

    # transformation
    df[col_name] = df[col_name]/100
    df[col_name] = df[col_name]+1
    df[col_name] = df[col_name]*init_value

    # concatenate two dataframe
    df2 = pd.concat([line, df])

    return df2


class PyKalman_PairTradingStrategy(bt.Strategy):

    params = (
        ('y_ticker', None),
        ('x_ticker', None),
        ('state_mean', None),
        ('delta', 0.0001),
        ('n_dim_state', None),
        ('Ve', None),
        ('kalman_averaging', False),
        ('entry_sqrt_q_multiplier', 1), # between 1 and 1.5
        ('exit_sqrt_q_multiplier', 0), # between 0 and 1
        ('initialisation_period', 30),
        ('risk', 1)
    )

    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def get_optimal_sizing(self, prices: np.object, betas: np.object, round_position=True):
        '''Get the optimal unit size'''

        # For now assume use (risk)% of portfolio for trades, (1-risk)% for re-balancing - risk in params
        if self.pos is not None:
            # use the previous size
            return self.optimal_size

        elif self.pos is None:
            trade_cash = self.broker.get_cash()*self.params.risk if self.broker.get_cash() > 0 else 0
            unit_price = prices.dot(abs(betas)) # assumes prices are all positive and not trading shorts on margin (i.e. hold full amt)
            size = trade_cash/unit_price

            if round_position:
                size = int(floor(size))

        return size

    def ord_tgt_size(self, current_size, current_price, target_value, round_position=False):
        ''' Calculates the target buy amount or sell size'''
        current_value = current_size*current_price
        val_diff = target_value - current_value
        tgt_size = val_diff/current_price

        if round_position:
            tgt_size = int(round(tgt_size))

        return tgt_size

    def get_pos_value(self, ticker, current_size, current_price, yesterdays_price):
        '''Returns the value of the position'''

        self.log(
            'Current cash ($): %.2f, Ticker %s - Position (units): %.2f, Previous value($): %.2f, Todays value ($): %.2f' %
            (self.broker.get_cash(),
             ticker,
             current_size,
             current_size*yesterdays_price,
             current_size*current_price))

    def notify_order(self, order):
        if order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            self.bar_executed[order.ref] = order
            return  # Await further notifications

        if order.status == order.Completed:
            self.bar_executed.pop(order.ref)
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
            else:
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

        elif order.status in [order.Margin]:

            self.bar_executed.pop(order.ref)
            self.log('%s ,' % order.Status[order.status])
            self.log('%s ,' % 'Closing all open positions')

            self.close_pos = True

        elif order.status in [order.Expired, order.Canceled]:

            self.bar_executed.pop(order.ref)
            self.log('%s ,' % order.Status[order.status])

            self.close_pos = True

        # Set notify pos to true
        self.notify_pos = True

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('%s OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.data._name, trade.pnl, trade.pnlcomm))

    def __init__(self):
        # To control operation entries
        self.bar_executed = {}
        self.close_pos = False
        self.optimal_size = None # To calculate our sizing
        self.pos = None # To check if we are long or short
        self.notify_pos = True # To check if there is a change for notification
        self.dataclose = self.datas[0].close

        # Kalman filter init
        trans_cov = self.params.delta / (1 - self.params.delta) * np.eye(self.params.n_dim_state)

        self.kf = MyKalmanPython(init_state_mean=self.params.state_mean,
                                 Ve=self.params.Ve, delta=self.params.delta, n_dim_state=self.params.n_dim_state,
                                 use_kalman_price_averaging=self.params.kalman_averaging)


    def next(self):

        # compute the kalman updates
        # check that y_ticker has the same name as data
        assert self.datas[0]._name == self.params.y_ticker
        y_mat = self.datas[0].close[0]

        x_mat = np.ones((self.params.n_dim_state,))
        for i in range(len(self.params.x_ticker)):
            x_ticker = self.params.x_ticker[i]

            # check that x_ticker has the same name as data
            assert self.datas[i+1]._name == x_ticker
            x_mat[i] = self.datas[i+1].close[0]

        # update the kalman filter
        obs_mat = np.asarray([x_mat])
        e, sqrt_Q, state_mean_ = self.kf.update(y_mat=y_mat,
                                                x_mat=obs_mat)

        if len(self) < self.params.initialisation_period:
            return  # return if still in initialisation period

        if len(self.bar_executed.keys()) > 0:
            return  # if an order is active, no new orders are allowed

        if self.close_pos:

            # close all open positions
            # (triggered when margin calls cancel orders or any orders are cancelled or expired)
            self.close(data=self.datas[0])
            for i in range(len(self.params.x_ticker)):
                x_ticker = self.params.x_ticker[i]
                self.close(data=self.datas[i + 1])
            self.pos = None

            # set flag to false
            self.close_pos = False
            return

        if self.notify_pos:
            # print position
            self.get_pos_value(ticker=self.params.y_ticker,
                               current_size=self.getposition(data=self.datas[0]).size,
                               current_price=self.datas[0].close[0],
                               yesterdays_price=self.datas[0].close[-1])

            for i in range(len(self.params.x_ticker)):
                x_ticker = self.params.x_ticker[i]
                self.get_pos_value(ticker=x_ticker,
                                   current_size=self.getposition(data=self.datas[i + 1]).size,
                                   current_price=self.datas[i + 1].close[0],
                                   yesterdays_price=self.datas[i + 1].close[-1])

            self.notify_pos = False

        # trading logic - close open positions first and re-balance portfolio
        if self.pos is not None:

            if self.pos == 'long' and e >= -(sqrt_Q*self.params.exit_sqrt_q_multiplier):

                # close your position
                self.close(data=self.datas[0])
                for i in range(len(self.params.x_ticker)):
                    self.close(data=self.datas[i + 1])
                self.pos = None

            elif self.pos == 'short' and e <= (sqrt_Q*self.params.exit_sqrt_q_multiplier):

                # close your position
                self.close(data=self.datas[0])
                for i in range(len(self.params.x_ticker)):
                    self.close(data=self.datas[i + 1])
                self.pos = None

            # else re-balance portfolio
            elif self.pos == 'long':

                # short x (only need to rebalance x)
                for i in range(len(self.params.x_ticker)):
                    value = -self.optimal_size * state_mean_[i] * self.datas[i + 1].close[0]
                    size = self.ord_tgt_size(current_size=self.getposition(data=self.datas[i + 1]).size,
                                             current_price=self.datas[i + 1].close[0],
                                             target_value=value)

                    if size > 0:
                        self.buy(data=self.datas[i + 1], size=size)
                    else:
                        self.sell(data=self.datas[i + 1], size=size)

            elif self.pos == 'short':

                # long x (only need to rebalance x)
                for i in range(len(self.params.x_ticker)):
                    value = self.optimal_size * state_mean_[i] * self.datas[i + 1].close[0]
                    size = self.ord_tgt_size(current_size=self.getposition(data=self.datas[i + 1]).size,
                                             current_price=self.datas[i + 1].close[0],
                                             target_value=value)

                    if size > 0:
                        self.buy(data=self.datas[i + 1], size=size)
                    else:
                        self.sell(data=self.datas[i + 1], size=size)

        if self.pos is None:

            if e < -(sqrt_Q*self.params.entry_sqrt_q_multiplier):

                # calculate size
                prices_mat = np.zeros(self.params.n_dim_state)
                betas_mat = np.ones(self.params.n_dim_state)

                prices_mat[0] = self.datas[0].close[0]
                betas_mat[0] = 1

                for i in range(len(self.params.x_ticker)):

                    prices_mat[i+1] = self.datas[i + 1].close[0]
                    betas_mat[i+1] = -1*state_mean_[i]

                self.optimal_size = self.get_optimal_sizing(prices=prices_mat,
                                                            betas=betas_mat)

                # long y and short x
                self.buy(data=self.datas[0], size=self.optimal_size)

                for i in range(len(self.params.x_ticker)):
                    value = -self.optimal_size * state_mean_[i] * self.datas[i + 1].close[0]
                    size = self.ord_tgt_size(current_size=0,
                                             current_price=self.datas[i + 1].close[0],
                                             target_value=value)

                    if size > 0:
                        self.buy(data=self.datas[i + 1], size=size)
                    else:
                        self.sell(data=self.datas[i + 1], size=size)

                self.pos = 'long'

            elif e > (sqrt_Q*self.params.entry_sqrt_q_multiplier):

                # calculate size
                prices_mat = np.zeros(self.params.n_dim_state)
                betas_mat = np.ones(self.params.n_dim_state)

                prices_mat[0] = self.datas[0].close[0]
                betas_mat[0] = -1

                for i in range(len(self.params.x_ticker)):
                    prices_mat[i + 1] = self.datas[i + 1].close[0]
                    betas_mat[i + 1] = state_mean_[i]

                self.optimal_size = self.get_optimal_sizing(prices=prices_mat,
                                                            betas=betas_mat)

                # short y and long x
                self.sell(data=self.datas[0], size=-self.optimal_size)

                for i in range(len(self.params.x_ticker)):
                    value = self.optimal_size * state_mean_[i] * self.datas[i + 1].close[0]
                    size = self.ord_tgt_size(current_size=0,
                                             current_price=self.datas[i + 1].close[0],
                                             target_value=value)

                    if size > 0:
                        self.buy(data=self.datas[i + 1], size=size)
                    else:
                        self.sell(data=self.datas[i + 1], size=size)

                self.pos = 'short'

        return


