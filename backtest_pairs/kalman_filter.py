# Kalman filter to estimate dynamic portfolio weights
from backtest_pairs.process_pairs import import_ticker_data, create_valid_ticker_combs
from backtest_pairs.utils import save_file_path
from pykalman import KalmanFilter
from backtest_pairs.hurst_half_life import half_life
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import sys
from math import ceil, sqrt
import statsmodels.tsa.stattools as ts


pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

etf_ticker_path = '/backtest_pairs/data/etf_tickers_07_2020.csv'
start_date = '2018-06-01'
end_date = '2020-12-01'
time_interval = 'weekly'
time_zones = [-18000, 0]
num_tickers_in_basket = 2
min_period_yrs = 1.5
max_half_life = 12 # in time interval units, 30 if days, 12 if weeks
min_half_life = 2 # in time interval units, 2 is default
use_close_prices = True

class MyKalmanPython:

    averaging_init_state_means = 0
    averaging_init_state_covariances = 1
    averaging_init_observation_covariances = 1
    averaging_init_transition_covariances = .01

    def __init__(self,
                 init_state_mean: np.object,
                 Ve: float,
                 delta: float,
                 n_dim_state: int,
                 use_kalman_price_averaging=False):
        """Init the Kalman filter variables"""

        # Kalman filter init
        self.Vw = delta / (1 - delta) * np.eye(n_dim_state) # delta=0.0001
        self.Ve = Ve # 0.001
        self.n_dim_state = n_dim_state

        self.R = np.zeros((n_dim_state, n_dim_state), dtype=np.float64 )
        self.P = np.zeros((n_dim_state, n_dim_state), dtype=np.float64 )

        # set beta
        self.beta = np.zeros((n_dim_state, 1))
        self.beta[:, 0] = init_state_mean

        # set yhat, e, q, and sqrt q
        self.yhat = np.zeros(1)  # measurement prediction
        self.e = np.zeros(1)
        self.Q = np.zeros(1)
        self.sqrt_Q = np.zeros(1)

        # set xmat
        self.x_mat = np.zeros((1, n_dim_state))

        # set ymat
        self.y_mat = np.zeros(1)

        # set timestep
        self.timestep = 0

        if use_kalman_price_averaging:

            self.kf_averaging = []
            self.kf_averaging_state_mean = []
            self.kf_averaging_state_covariance = []
            for i in range(n_dim_state):

                # Construct a Kalman filter for each price series, there are n_dim_state of them
                self.kf_averaging[i] = KalmanFilter(transition_matrices=[1],
                                                    observation_matrices=[1],
                                                    initial_state_mean=self.averaging_init_state_means,
                                                    initial_state_covariance=self.averaging_init_state_covariances,
                                                    observation_covariance=self.averaging_init_observation_covariances,
                                                    transition_covariance=[self.averaging_init_transition_covariances])
        else:

            self.kf_averaging = None

    def update(self,
               x_mat: np.object,
               y_mat: float):
        """Update the Kalman filter"""

        if self.kf_averaging:

            # y mat first
            obs_mat = np.asarray([y_mat])
            state_mean_, state_covariance_ = self.kf_averaging[0].filter_update(self.state_mean,
                                                                                self.state_covariance,
                                                                                observation=y_mat,
                                                                                observation_matrix=obs_mat)

            x_mat = np.asarray([state_mean_])

        # update the kalman filter
        if self.timestep > 0:
            beta_ = self.beta[:, self.timestep-1]
            self.R = self.P + self.Vw
        else:
            beta_ = self.beta[:, 0]

        yhat = np.dot(x_mat, beta_)
        Q = np.dot(np.dot(x_mat, self.R), x_mat.T) + self.Ve
        sqrt_Q = np.sqrt(Q)

        # Observe y(t)
        e = y_mat - yhat  # measurement prediction error
        K = np.dot(self.R, x_mat.T) / Q  # Kalman gain
        beta_ = beta_ + np.dot(K, e)  # State update. Equation 3.11
        self.P = self.R - np.dot(np.dot(K.squeeze(), x_mat.squeeze(0).T), self.R)  # State covariance update. Euqation 3.12

        # update beta, yhat, q, sqrt_q
        if self.timestep == 0:
            self.beta[:, self.timestep] = beta_
            self.yhat[self.timestep] = yhat
            self.e[self.timestep] = e
            self.Q[self.timestep] = Q
            self.sqrt_Q[self.timestep] = sqrt_Q
            self.x_mat[self.timestep, :] = x_mat
            self.y_mat[self.timestep] = y_mat
        else:
            beta_shape = np.zeros((2, 1))
            beta_shape[:, 0] = beta_
            self.beta = np.hstack((self.beta, beta_shape))
            self.yhat = np.hstack((self.yhat, yhat))
            self.e = np.hstack((self.e, e))
            self.Q = np.hstack((self.Q, Q.squeeze(0)))
            self.sqrt_Q = np.hstack((self.sqrt_Q, sqrt_Q.squeeze(0)))
            self.x_mat = np.vstack((self.x_mat, x_mat))
            self.y_mat = np.hstack((self.y_mat, y_mat))

        # update timestep
        self.timestep += 1

        return e, sqrt_Q, beta_


    def reset(self):

        # Reset values
        self.R = np.zeros((self.n_dim_state, self.n_dim_state))
        self.P = np.zeros((self.n_dim_state, self.n_dim_state))

        # set beta
        self.beta = np.zeros((self.n_dim_state, 1))

        # set yhat, e, q, and sqrt q
        self.yhat = np.zeros(1)  # measurement prediction
        self.e = np.zeros(1)
        self.Q = np.zeros(1)
        self.sqrt_Q = np.zeros(1)

        # set xmat
        self.x_mat = np.zeros((1, self.n_dim_state))

        # set ymat
        self.y_mat = np.zeros(1)

        # set timestep
        self.timestep = 0

        if self.kf_averaging:

            self.kf_averaging = []
            self.kf_averaging_state_mean = []
            self.kf_averaging_state_covariance = []
            for i in self.n_dim_state:
                # Construct a Kalman filter for each price series, there are n_dim_state of them
                self.kf_averaging[i] = KalmanFilter(transition_matrices=[1],
                                                    observation_matrices=[1],
                                                    initial_state_mean=self.params.averaging_init_state_means,
                                                    initial_state_covariance=self.params.averaging_init_state_covariances,
                                                    observation_covariance=self.params.averaging_init_observation_covariances,
                                                    transition_covariance=self.params.averaging_init_transition_covariances)
        else:

            self.kf_averaging = None

    def apply_Kalman_average(self, x):

        # Construct a Kalman filter
        kf = KalmanFilter(transition_matrices=[1],
                          observation_matrices=[1],
                          initial_state_mean=0,
                          initial_state_covariance=1,
                          observation_covariance=1,
                          transition_covariance=.01)

        # Use the observed values of the price to get a rolling mean
        state_means, _ = kf.filter(x.values)
        state_means = pd.Series(state_means.flatten(), index=x.index)

        return state_means


def reformat_data_y_x_tickers(valid_combinations: pd.Series, zero_mean=False):

    # get the Johansen weights
    p_weights = valid_combinations['johansen_eigenvectors']
    tickers = valid_combinations['ticker']

    # normalise the weights so that the first element is 1
    if isinstance(p_weights, list):
        p_weights = pd.Series(p_weights)

    scale = 1/p_weights.iloc[0]
    p_weights = p_weights*scale*-1

    # y is the first security value, x is the remaining
    y_ticker = tickers[0]
    x_ticker = tickers[1:]
    initial_mean = p_weights[1:].to_list() + [0] # 0 is for the constant

    if zero_mean:
        initial_mean = [0 for x in initial_mean]

    return y_ticker, x_ticker, initial_mean, len(x_ticker) + 1


def reformat_data(df: pd.DataFrame, zero_mean=False):

    # create a series and get the number of ticker combinations
    ticker_series = pd.Series()
    sample_ticker = df['ticker'][0]
    n_combs = len(sample_ticker)

    def shift(seq, n=0):
        a = n % len(seq)
        return seq[-a:] + seq[:-a]

    for i in range(n_combs):

        ticker_series = pd.concat([ticker_series, df['ticker'].apply(shift, args=(i,))], ignore_index=False)

    # join s1 with df on their indexes
    df = pd.merge(df, ticker_series.to_frame('updated_ticker'), left_index=True, right_index=True, how="inner")

    # drop ticker col and rename updated_ticker to ticker
    df['ticker'] = df['updated_ticker']
    df = df.drop('updated_ticker', 1)

    # reset index
    df.reset_index(drop=True, inplace=True)

    # get y and x tickers
    df['y_ticker'], df['x_ticker'], df['initial_mean'], df['dim'] = zip(*df.apply(reformat_data_y_x_tickers, args=(zero_mean,), axis=1))

    return df


def apply_active_Kalman_filter(valid_combinations: pd.Series, price_averaging=False, min_date: str = None, max_date: str = None):

    y_series = valid_combinations['merged_prices_comb_df'][valid_combinations['y_ticker']]
    x_df = valid_combinations['merged_prices_comb_df'][list(valid_combinations['x_ticker'])]

    if min_date:

        # cap y_series and x_df at min date
        y_series = y_series.loc[min_date:]
        x_df = x_df.loc[min_date:]

    if max_date:

        # cap y_series and x_df at min date
        y_series = y_series.loc[:max_date]
        x_df = x_df.loc[:max_date]

    init_state_mean = valid_combinations['initial_mean']
    dim = valid_combinations['dim']

    x_mat = np.ones((y_series.shape[0], dim))
    for i in range(len(valid_combinations['x_ticker'])):
        x_ticker = valid_combinations['x_ticker'][i]
        x_mat[:, i] = (x_df[x_ticker])

    kf = MyKalmanPython(init_state_mean=init_state_mean,
                        Ve=0.001, delta=0.0001, n_dim_state=dim, use_kalman_price_averaging=price_averaging)

    for idx in range(0, y_series.shape[0]):

        obs_mat = np.asarray([x_mat[idx, :]])
        kf.update(y_mat=y_series.iloc[idx],
                  x_mat=obs_mat)


    beta_cols = list(valid_combinations['x_ticker'])
    beta_cols = beta_cols + ['const']
    pre_str = '_beta'
    beta_cols = [s + pre_str for s in beta_cols]
    beta_weights_df = pd.DataFrame(data=kf.beta.T,
                                   index=x_df.index.to_list(),
                                   columns=beta_cols)
    kf_soln = pd.merge(x_df, beta_weights_df, how='inner', left_index=True, right_index=True)
    kf_soln = pd.merge(kf_soln, y_series, how='inner', left_index=True, right_index=True)

    kf_soln['y_hat'] = pd.Series(kf.yhat, index=kf_soln.index)
    kf_soln['e'] = pd.Series(kf.e, index=kf_soln.index)
    kf_soln['q'] = pd.Series(kf.Q, index=kf_soln.index)
    kf_soln['sqrt_q'] = pd.Series(kf.sqrt_Q, index=kf_soln.index)

    # estimate returns
    APR, Sharpe = estimate_returns(kf_soln, beta_cols, valid_combinations['y_ticker'],
                                   list(valid_combinations['x_ticker']),
                                   dim, 'kalman_raw')

    return APR, Sharpe


def apply_raw_Kalman_filter(valid_combinations: pd.Series, plot=False, min_date: str = None, max_date: str = None):

    y_series = valid_combinations['merged_prices_comb_df'][valid_combinations['y_ticker']]
    x_df = valid_combinations['merged_prices_comb_df'][list(valid_combinations['x_ticker'])]

    if min_date:

        # cap y_series and x_df at min date
        y_series = y_series.loc[min_date:]
        x_df = x_df.loc[min_date:]

    if max_date:

        # cap y_series and x_df at min date
        y_series = y_series.loc[:max_date]
        x_df = x_df.loc[:max_date]

    init_state_mean = valid_combinations['initial_mean']
    dim = valid_combinations['dim']

    delta = 0.0001  # delta=1 gives fastest change in beta, delta=0.000....1 allows no change (like traditional linear regression).
    R = np.zeros((dim, dim), dtype=np.float64)
    P = np.zeros((dim, dim), dtype=np.float64)
    beta = np.full((dim, x_df.shape[0]), np.nan)
    Vw = delta / (1 - delta) * np.eye(dim)
    Ve = 0.001
    yhat = np.full(y_series.shape[0], np.nan)  # measurement prediction
    e = yhat.copy()
    Q = yhat.copy()
    sqrt_Q = yhat.copy()

    x_mat = np.ones((y_series.shape[0], dim))

    # Compute a smoothed y, x signal
    # state_means = KalmanFilterRegression(apply_Kalman_average(x), apply_Kalman_average(y))

    for i in range(len(valid_combinations['x_ticker'])):
        x_ticker = valid_combinations['x_ticker'][i]
        x_mat[:, i] = (x_df[x_ticker])

    # Initialize beta(:, 0) to zero
    beta[:, 0] = init_state_mean

    # Given initial beta and R (and P)
    for t in range(y_series.shape[0]):
        if t > 0:
            beta[:, t] = beta[:, t - 1]
            R = P + Vw

        yhat[t] = np.dot(x_mat[t, :], beta[:, t])
        #    print('FIRST: yhat[t]=', yhat[t])

        Q[t] = np.dot(np.dot(x_mat[t, :], R), x_mat[t, :].T) + Ve
        #    print('Q[t]=', Q[t])

        sqrt_Q[t] = np.sqrt(Q[t])

        # Observe y(t)
        e[t] = y_series[t] - yhat[t]  # measurement prediction error
        #    print('e[t]=', e[t])
        #    print('SECOND: yhat[t]=', yhat[t])

        K = np.dot(R, x_mat[t, :].T) / Q[t]  # Kalman gain
        #    print(K)

        beta[:, t] = beta[:, t] + np.dot(K, e[t])  # State update. Equation 3.11
        #    print(beta[:, t])


        P = R - np.dot(np.dot(K, x_mat[t, :]), R)  # State covariance update. Euqation 3.12
    #    print(R)

    beta_cols = list(valid_combinations['x_ticker'])
    beta_cols = beta_cols + ['const']
    pre_str = '_beta'
    beta_cols = [s + pre_str for s in beta_cols]
    beta_weights_df = pd.DataFrame(data=beta.T,
                                   index=x_df.index.to_list(),
                                   columns=beta_cols)
    kf_soln = pd.merge(x_df, beta_weights_df, how='inner', left_index=True, right_index=True)
    kf_soln = pd.merge(kf_soln, y_series, how='inner', left_index=True, right_index=True)

    kf_soln['y_hat'] = pd.Series(yhat, index=kf_soln.index)
    kf_soln['e'] = pd.Series(e, index=kf_soln.index)
    kf_soln['q'] = pd.Series(Q, index=kf_soln.index)
    kf_soln['sqrt_q'] = pd.Series(sqrt_Q, index=kf_soln.index)

    # plot figure
    if plot:
        plt_cols = [['e', 'sqrt_q']] + \
                   np.expand_dims(beta_cols, axis=1).tolist()
        fig, axs = plt.subplots(nrows=2, ncols=ceil(len(plt_cols) / 2))
        for ax, col in zip(axs.flatten(), plt_cols):
            sns.lineplot(data=kf_soln[col], ax=ax)

        fig.savefig(
            save_file_path('plots',
                           'kalman_raw_y_' + valid_combinations['y_ticker'] + '_x_' + '_'.join(beta_cols) + '.png'),
            format='png', dpi=1000)
        fig.clf()

    # estimate returns
    APR, Sharpe = estimate_returns(kf_soln, beta_cols, valid_combinations['y_ticker'], list(valid_combinations['x_ticker']),
                                   dim, 'kalman_raw', plot=plot)

    return APR, Sharpe


def estimate_returns(frame: pd.DataFrame, beta_cols: list,
                     y_ticker: str, x_cols: list, dim: int, run_name: str, initialisation_period=30,
                     plot=False):

    # Estimate returns excluding transaction costs and slippage
    frame['longsEntry'] = frame.e < -frame.sqrt_q
    frame['longsExit'] = frame.e > 0

    frame['shortsEntry'] = frame.e > frame.sqrt_q
    frame['shortsExit'] = frame.e < 0

    # set the first initialisation_period rows to zero for long entry, exit and short entry, exit
    frame.iloc[:initialisation_period, frame.columns.get_loc('longsEntry')] = False
    frame.iloc[:initialisation_period, frame.columns.get_loc('longsExit')] = False
    frame.iloc[:initialisation_period, frame.columns.get_loc('shortsEntry')] = False
    frame.iloc[:initialisation_period, frame.columns.get_loc('shortsExit')] = False

    numUnitsLong = np.zeros(frame.longsEntry.shape)
    numUnitsLong[:] = np.nan

    numUnitsShort = np.zeros(frame.shortsEntry.shape)
    numUnitsShort[:] = np.nan

    numUnitsLong[0] = 0
    numUnitsLong[frame.longsEntry] = 1
    numUnitsLong[frame.longsExit] = 0
    numUnitsLong = pd.DataFrame(numUnitsLong)
    numUnitsLong.fillna(method='ffill', inplace=True)

    numUnitsShort[0] = 0
    numUnitsShort[frame.shortsEntry] = -1
    numUnitsShort[frame.shortsExit] = 0
    numUnitsShort = pd.DataFrame(numUnitsShort)
    numUnitsShort.fillna(method='ffill', inplace=True)

    # exclude const from beta cols
    beta_cols_tickers = beta_cols.copy()
    beta_cols_tickers.remove('const_beta')
    y_x_cols = [y_ticker] + x_cols

    numUnits = numUnitsLong + numUnitsShort
    positions = pd.DataFrame(np.tile(numUnits.values, [1, dim]) * ts.add_constant(-np.asarray(frame[beta_cols_tickers])) * frame[y_x_cols].values)
    # [hedgeRatio -ones(size(hedgeRatio))] is the shares allocation, [hedgeRatio -ones(size(hedgeRatio))].*y2 is the dollar capital allocation, while positions is the dollar capital in each ETF.

    pnl = np.sum((positions.shift().values) * (frame[y_x_cols].pct_change().values), axis=1)  # daily P&L of the strategy
    ret = pnl / np.sum(np.abs(positions.shift()), axis=1)
    cumreturn = np.cumprod(1 + ret) - 1

    # plot figure
    if plot:
        sns_plot = sns.lineplot(data=cumreturn)
        sns_plot.figure.savefig(
            save_file_path('plots',
                           run_name + '_cum_return_y_' + y_ticker + '_x_' + '_'.join(beta_cols) + '.png'),
            format='png', dpi=1000)
        sns_plot.figure.clf()

    # (np.cumprod(1 + ret) - 1).plot()
    APR = np.prod(1 + ret) ** (252 / len(ret)) - 1
    Sharpe = np.sqrt(252) * np.mean(ret) / np.std(ret)
    print('Processed ticker: {a}, {b} - APR={c}, Sharpe={d}'.format(a=y_ticker, b=x_cols, c=APR, d=Sharpe))

    return APR, Sharpe


if __name__== '__main__':

    # download and import data
    print('Importing ticker data')
    ticker_data = import_ticker_data(tickers=['IXJ', 'TMF'],
                                     start_date=start_date,
                                     end_date=end_date,
                                     time_interval='daily')

    # calculating valid ticker combinations
    print('Calculating valid ticker combinations')
    valid_combinations = create_valid_ticker_combs(ticker_data, min_period_yrs=min_period_yrs,
                                                   num_tickers_in_basket=num_tickers_in_basket,
                                                   max_half_life=max_half_life, min_half_life=min_half_life,
                                                   time_zones=time_zones, save_all=True, time_interval=time_interval,
                                                   use_close_prices=use_close_prices)

    if valid_combinations.shape[0] == 0:
        warnings.warn('No valid ticker combinations to process!')
        sys.exit(0)

    # Filter only on valid combinations
    print('Filtering valid ticker combinations only')
    valid_combinations = valid_combinations.loc[valid_combinations['sample_pass'] == True]

    # # Load valid combination from file
    # print('Loading valid ticker combinations')
    # valid_combinations = pd.read_pickle("C:\\Users\\hamzajuzer\\Documents\\Algorithmic Trading\\AlgoTradingv1\\backtest_pairs\\coint_results\\valid_coint_results_df_12_2020_weekly.pkl")

    if valid_combinations.shape[0] == 0:
        warnings.warn('No valid ticker combinations to process!')
        sys.exit(0)

    # For each valid combination
    print('Reformatting valid ticker combination data')
    valid_combinations = reformat_data(valid_combinations, zero_mean=False)

    # # Test Kalman Filter active code on ticker
    # print('Applying online Kalman Filter on valid ticker combination data')
    # valid_combinations['APR_active'], valid_combinations['Sharpe_active'] = zip(*valid_combinations.apply(apply_active_Kalman_filter, args=(False, None, end_date), axis=1))

    # Test Kalman Filter active code with price averaging on ticker
    # print('Applying online Kalman Filter with price averaging on valid ticker combination data')
    # valid_combinations['APR_active_pa'], valid_combinations['Sharpe_active_pa'] = zip(
    #     *valid_combinations.apply(apply_active_Kalman_filter, args=(True, None, None), axis=1))

    print('Applying raw Kalman Filter on valid ticker combination data')
    valid_combinations['APR'], valid_combinations['Sharpe'] = zip(*valid_combinations.apply(apply_raw_Kalman_filter, args=(False, None, end_date), axis=1))

    # Save dataframe
    # print('Saving results')
    # valid_combinations.to_pickle(
    #     save_file_path(folder_name='coint_results', filename='kalman_results_df_12_2020_weekly.pkl'))

