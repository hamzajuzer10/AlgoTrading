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

etf_ticker_path = '/backtest_pairs/data/etf_tickers.csv'
start_date = '2006-04-26'
end_date = '2012-04-09'
time_interval = 'daily'
num_tickers_in_basket = 2
min_period_yrs = 3
max_half_life = 30 # in time interval units
min_half_life = 0 # in time interval units


class MyKalmanPython:

    params = {
        ('averaging_init_state_means', 0),
        ('averaging_init_state_covariances', 1),
        ('averaging_init_observation_covariances', 1),
        ('averaging_init_transition_covariances', .01)
    }

    def __init__(self,
                 init_state_mean: np.object,
                 init_state_covariance: np.object,
                 delta: float,
                 n_dim_state: int,
                 observation_covariance: float,
                 use_kalman_price_averaging=False):
        """Init the Kalman filter variables"""

        # Kalman filter init
        trans_cov = delta / (1 - delta) * np.eye(n_dim_state)

        self.kf = KalmanFilter(n_dim_obs=1, n_dim_state=n_dim_state,
                               transition_matrices=np.eye(n_dim_state),
                               observation_covariance=observation_covariance,
                               transition_covariance=trans_cov)

        self.state_mean = init_state_mean
        self.state_covariance = init_state_covariance

        if use_kalman_price_averaging:

            self.kf_averaging = []
            self.kf_averaging_state_mean = []
            self.kf_averaging_state_covariance = []
            for i in n_dim_state:

                # Construct a Kalman filter for each price series, there are n_dim_state of them
                self.kf_averaging[i] = KalmanFilter(transition_matrices=[1],
                                                    observation_matrices=[1],
                                                    initial_state_mean=self.params.averaging_init_state_means,
                                                    initial_state_covariance=self.params.averaging_init_state_covariances,
                                                    observation_covariance=self.params.averaging_init_observation_covariances,
                                                    transition_covariance=self.params.averaging_init_transition_covariances)
        else:

            self.kf_averaging = None

    def update(self,
               x_mat: np.object,
               y_mat: np.object):
        """Update the Kalman filter"""

        if self.kf_averaging:

            # y mat first
            obs_mat = np.asarray([y_mat])
            state_mean_, state_covariance_ = self.kf_averaging[0].filter_update(self.state_mean,
                                                                                self.state_covariance,
                                                                                observation=y_mat,
                                                                                observation_matrix=obs_mat)


        # update the kalman filter
        obs_mat = np.asarray([x_mat])
        state_mean_, state_covariance_ = self.kf.filter_update(self.state_mean,
                                                               self.state_covariance,
                                                               observation=y_mat,
                                                               observation_matrix=obs_mat)
        self.state_mean = np.asarray(state_mean_)
        self.state_covariance = np.asarray(state_covariance_)

        Q = obs_mat.dot(np.asarray(state_covariance_)).dot(obs_mat.T)
        sqrt_Q = sqrt(Q.item())

        # compute spread, e
        e = y_mat.item() - x_mat.dot(state_mean_)

        return e, sqrt_Q

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


def reformat_data(valid_combinations: pd.Series):

    # get the Johansen weights
    p_weights = valid_combinations['johansen_eigenvectors']

    # normalise the weights so that the first element is 1
    scale = 1/p_weights.iloc[0]
    p_weights = p_weights*scale*-1

    # y is the first security value, x is the remaining
    y_ticker = p_weights.index[0]
    x_ticker = p_weights.index[1:].to_list()
    initial_const = valid_combinations['merged_prices_comb_df']['portfolio_price'].mean()
    initial_mean = p_weights[1:].to_list() + [0]

    return y_ticker, x_ticker, initial_mean, len(x_ticker) + 1


def apply_active_Kalman_filter(valid_combinations: pd.Series):
    y_series = valid_combinations['merged_prices_comb_df'][valid_combinations['y_ticker']]
    x_df = valid_combinations['merged_prices_comb_df'][list(valid_combinations['x_ticker'])]
    init_state_mean = valid_combinations['initial_mean']
    dim = valid_combinations['dim']

    delta = 0.0001
    observation_covariance = 0.05
    trans_cov = delta / (1 - delta) * np.eye(dim)

    x_mat = np.ones((y_series.shape[0], dim))
    for i in range(len(valid_combinations['x_ticker'])):
        x_ticker = valid_combinations['x_ticker'][i]
        x_mat[:, i] = (x_df[x_ticker])

    # Compute a smoothed y, x signal
    # state_means = KalmanFilterRegression(apply_Kalman_average(x), apply_Kalman_average(y))

    state_means = np.zeros((y_series.shape[0]+1, dim))
    state_covs = np.zeros((y_series.shape[0]+1, dim, dim))
    Q = np.zeros((y_series.shape[0]+1,))
    sqrt_Q = np.zeros((y_series.shape[0]+1,))

    # initialize state means and covariances
    state_means[0,:] = init_state_mean
    state_covs[0, :, :] = np.zeros((dim, dim))
    Q[0] = 0
    sqrt_Q[0] = 0

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=dim,
                      transition_matrices=np.eye(dim),
                      observation_covariance=observation_covariance,
                      transition_covariance=trans_cov)

    for idx in range(0, y_series.shape[0]):

        obs_mat = np.asarray([x_mat[idx, :]])
        state_means[idx+1, :], state_covs[idx+1, :, :] = kf.filter_update(state_means[idx, :],
                                                                          state_covs[idx, :, :],
                                                                          observation=np.asarray(y_series.iloc[idx]),
                                                                          observation_matrix=obs_mat)

        Q[idx+1] = obs_mat.dot(state_covs[idx+1, :, :]).dot(obs_mat.T)
        sqrt_Q[idx+1] = np.sqrt(Q[idx+1])

    # remove all index 0s from state_means, state_covs, Q, sqrt_Q
    state_means = np.delete(state_means, (0), axis=0)
    Q = np.delete(Q, (0), axis=0)
    sqrt_Q = np.delete(sqrt_Q, (0), axis=0)

    # plot and compare errors
    beta_cols = list(valid_combinations['x_ticker'])
    beta_cols = beta_cols + ['const']
    pre_str = '_beta'
    beta_cols = [s + pre_str for s in beta_cols]
    beta_weights_df = pd.DataFrame(data=state_means,
                                   index=x_df.index.to_list(),
                                   columns=beta_cols)
    kf_soln = pd.merge(x_df, beta_weights_df, how='inner', left_index=True, right_index=True)
    kf_soln = pd.merge(kf_soln, y_series, how='inner', left_index=True, right_index=True)

    # calculate y_hat and spread (e)
    kf_soln['y_hat'], kf_soln['e'] = zip(*kf_soln.apply(compute_y_hat_and_spread, args=(valid_combinations['y_ticker'],
                                                                                        valid_combinations['x_ticker'],), axis=1))

    # calculate half life
    half_life_e = half_life(kf_soln.e)

    sqrt_Q_df = pd.DataFrame(data={'q': Q, 'sqrt_q': sqrt_Q}, index=x_df.index.to_list(), columns=['q', 'sqrt_q'])
    kf_soln = pd.merge(kf_soln, sqrt_Q_df, how='inner', left_index=True, right_index=True)

    # get z score for comparison
    kf_soln['zScore'] = kf_soln[['e']].apply(apply_Z_score, args=(half_life_e, 2,))

    plt_cols = [['e', 'sqrt_q'], ['zScore']] + \
               np.expand_dims(beta_cols, axis=1).tolist()
    fig, axs = plt.subplots(nrows=2, ncols=ceil(len(plt_cols) / 2))
    for ax, col in zip(axs.flatten(), plt_cols):
        sns.lineplot(data=kf_soln[col], ax=ax)

    fig.savefig(
        save_file_path('plots', 'kalman_online_y_' + valid_combinations['y_ticker'] + '_x_' + '_'.join(beta_cols) + '.png'),
        format='png', dpi=1000)
    fig.clf()

    # estimate returns
    APR, Sharpe = estimate_returns(kf_soln, beta_cols, valid_combinations['y_ticker'], list(valid_combinations['x_ticker']),
                                   dim, 'kalman_online')

    return APR, Sharpe


def apply_raw_Kalman_filter(valid_combinations: pd.Series):

    y_series = valid_combinations['merged_prices_comb_df'][valid_combinations['y_ticker']]
    x_df = valid_combinations['merged_prices_comb_df'][list(valid_combinations['x_ticker'])]
    init_state_mean = valid_combinations['initial_mean']
    dim = valid_combinations['dim']

    delta = 0.0001  # delta=1 gives fastest change in beta, delta=0.000....1 allows no change (like traditional linear regression).
    R = np.zeros((dim, dim))
    P = R.copy()
    beta = np.full((dim, x_df.shape[0]), np.nan)
    Vw = delta / (1 - delta) * np.eye(2)
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
                                   dim, 'kalman_raw')

    return APR, Sharpe


def compute_y_hat_and_spread(row: pd.Series, y_ticker: str, x_cols: list):

    y_hat = 0
    if isinstance(x_cols, str): x_cols = [x_cols]
    for t in x_cols:

        b_t = t+'_beta'
        y_hat += row[t]*row[b_t]

    y_hat += row['const_beta']
    e = row[y_ticker] - y_hat

    return y_hat, e


def apply_Z_score(e: pd.Series, half_life: float, half_life_multiplier=1):

    # get z score for comparison
    mean_spread = e.rolling(window=ceil(half_life)*half_life_multiplier).mean()
    std_spread = e.rolling(window=ceil(half_life)*half_life_multiplier).std()

    z = (e - mean_spread) / std_spread

    return z


def estimate_returns(frame: pd.DataFrame, beta_cols: list,
                     y_ticker: str, x_cols: list, dim: int, run_name: str, initialisation_period=30):

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
    sns_plot= sns.lineplot(data=cumreturn)
    sns_plot.figure.savefig(
        save_file_path('plots',
                       run_name + '_cum_return_' + valid_combinations['y_ticker'].item() + '_x_' + '_'.join(beta_cols) + '.png'),
        format='png', dpi=1000)
    sns_plot.figure.clf()

    # (np.cumprod(1 + ret) - 1).plot()
    APR = np.prod(1 + ret) ** (252 / len(ret)) - 1
    Sharpe = np.sqrt(252) * np.mean(ret) / np.std(ret)
    print('APR=%f Sharpe=%f' % (APR, Sharpe))

    return APR, Sharpe


if __name__== '__main__':

    # download and import data
    print('Importing ticker data')
    ticker_data = import_ticker_data(ticker_file_path=etf_ticker_path,
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
    valid_combinations['initial_mean'], valid_combinations['dim'] = zip(*valid_combinations.apply(reformat_data, axis=1))

    # Test Kalman Filter active code on ticker
    print('Applying active Kalman Filter on valid ticker combination data')
    valid_combinations.apply(apply_active_Kalman_filter, axis=1)

    # print('Applying raw Kalman Filter on valid ticker combination data')
    # valid_combinations.apply(apply_raw_Kalman_filter, axis=1)

