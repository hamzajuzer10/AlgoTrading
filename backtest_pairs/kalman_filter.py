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
from math import ceil
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

etf_ticker_path = 'C:\\Users\\hamzajuzer\\Documents\\Algorithmic Trading\\AlgoTradingv1\\backtest_pairs\\etf_tickers.csv'
start_date = '2006-04-26'
end_date = '2012-04-09'
time_interval = 'daily'
num_tickers_in_basket = 2
min_period_yrs = 3
max_half_life = 30 # in time interval units
min_half_life = 0 # in time interval units


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
    initial_mean = p_weights[1:].to_list() + [initial_const]

    valid_combinations['y_ticker'] = y_ticker
    valid_combinations['x_ticker'] = x_ticker
    valid_combinations['initial_mean'] = initial_mean

    return valid_combinations


def apply_Kalman_filter(valid_combinations: pd.Series):

    y_series = valid_combinations['merged_prices_comb_df'][valid_combinations['y_ticker']]
    x_df = valid_combinations['merged_prices_comb_df'][valid_combinations['x_ticker']]
    init_state_mean = valid_combinations['initial_mean']
    dim = len(valid_combinations['x_ticker']) + 1

    delta = 0.0001
    trans_cov = delta / (1 - delta) * np.eye(dim)

    x_series = []
    for x_ticker in valid_combinations['x_ticker']:
        x_series.append(x_df[x_ticker])

    # Compute a smoothed y, x signal
    # state_means = KalmanFilterRegression(apply_Kalman_average(x), apply_Kalman_average(y))

    obs_mat = np.expand_dims(np.vstack([x_series, [np.ones(len(x_df))]]).T, axis=1)

    kf = KalmanFilter(n_dim_obs=1
                      , n_dim_state=dim # y is 1-dimensional, (alpha, beta_i) is n-dimensional
                      , initial_state_mean=init_state_mean
                      , initial_state_covariance=np.zeros((dim, dim)) # need to optimize
                      , transition_matrices=np.eye(dim)
                      , observation_matrices=obs_mat
                      , observation_covariance=0.05 # need to optimize
                      , transition_covariance=trans_cov)

    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.filter(y_series.values)

    # plot and compare errors
    beta_cols = valid_combinations['x_ticker']
    beta_cols = beta_cols + ['const']
    pre_str = '_beta'
    beta_cols = [s + pre_str for s in beta_cols]
    beta_weights_df = pd.DataFrame(data=state_means,
                                   index=x_df.index.to_list(),
                                   columns=beta_cols)
    kf_soln = pd.merge(x_df, beta_weights_df, how='inner', left_index=True, right_index=True)
    kf_soln = pd.merge(kf_soln, y_series, how='inner', left_index=True, right_index=True)

    # calculate y_hat and spread (e)
    kf_soln = kf_soln.apply(compute_y_hat_and_spread, args=(True, valid_combinations['y_ticker'],
                                                            valid_combinations['x_ticker'],), axis=1)

    # calculate y_hat and spread (e) without the constant
    kf_soln = kf_soln.apply(compute_y_hat_and_spread, args=(False, valid_combinations['y_ticker'],
                                                 valid_combinations['x_ticker'],), axis=1)

    # calculate half life
    half_life_e = half_life(kf_soln.e)
    half_life_e_exc_const = half_life(kf_soln.e_exc_const)

    # get Q and sqrt Q
    Q = np.zeros(y_series.shape)
    sqrt_Q = np.zeros(y_series.shape)
    for t in range(0, y_series.shape[0]):
        Q[t] = obs_mat[t, 0, :].dot(state_covs[t, :, :]).dot(obs_mat[t, 0, :].T) + 0.001
        sqrt_Q[t] = np.sqrt(Q[t])

    sqrt_Q_df = pd.DataFrame(data={'q': Q, 'sqrt_q': sqrt_Q}, index=x_df.index.to_list(), columns=['q', 'sqrt_q'])
    kf_soln = pd.merge(kf_soln, sqrt_Q_df, how='inner', left_index=True, right_index=True)

    # get z score for comparison
    kf_soln['zScore'] = kf_soln[['e']].apply(apply_Z_score, args=(half_life_e, 2,))
    kf_soln['zScore_exc_const'] = kf_soln[['e']].apply(apply_Z_score, args=(half_life_e_exc_const, 1,))

    plt_cols = [['sqrt_q', 'e'], ['zScore'], ['zScore_exc_const']] + \
               np.expand_dims(beta_cols, axis=1).tolist()
    fig, axs = plt.subplots(nrows=2, ncols=ceil(len(plt_cols)/2))
    for ax, col in zip(axs.flatten(), plt_cols):
        sns.lineplot(data=kf_soln[col], ax=ax)

    fig.savefig(save_file_path('plots', 'kalman_y_'+valid_combinations['y_ticker']+'_x_' + '_'.join(beta_cols) +'.png'),
                format='png', dpi=1000)
    fig.clf()

    return valid_combinations


def compute_y_hat_and_spread(row: pd.Series, const: bool, y_ticker: str, x_cols: list):

    y_hat=0
    if isinstance(x_cols, str): x_cols = [x_cols]
    for t in x_cols:

        b_t = t+'_beta'
        y_hat += row[t]*row[b_t]

    if const:
        y_hat += row['const_beta']
        field_name = 'y_hat'
        spread_name = 'e'
    else:
        field_name = 'y_hat_exc_const'
        spread_name = 'e_exc_const'

    row[field_name] = y_hat
    row[spread_name] = row[y_ticker] - row[field_name]

    return row


def apply_Z_score(e: pd.Series, half_life: float, half_life_multiplier=1):

    # get z score for comparison
    mean_spread = e.rolling(window=ceil(half_life)*half_life_multiplier).mean()
    std_spread = e.rolling(window=ceil(half_life)*half_life_multiplier).std()

    z = (e - mean_spread) / std_spread

    return z


def apply_Kalman_average(x):

  # Construct a Kalman filter
  kf = KalmanFilter(transition_matrices = [1],
     observation_matrices = [1],
     initial_state_mean = 0,
     initial_state_covariance = 1,
     observation_covariance=1,
     transition_covariance=.01)

  # Use the observed values of the price to get a rolling mean
  state_means, _ = kf.filter(x.values)
  state_means = pd.Series(state_means.flatten(), index=x.index)

  return state_means


def estimate_returns(frame: pd.DataFrame, sqrt_q_multiplier_signal=1):

    # Estimate returns excluding transaction costs and slippage

    # trading logic
    # set up num units long
    frame['long entry'] = (frame['e'] < - frame['sqrt_q'])
    frame['long exit'] = (frame['e'] > - frame['sqrt_q'])
    frame['num units long'] = np.nan
    frame.loc[frame['long entry'], 'num units long'] = 1
    frame.loc[frame['long exit'], 'num units long'] = 0
    frame['num units long'][0] = 0
    frame['num units long'] = frame['num units long'].fillna(
        method='pad')

    # set up num units short
    frame['short entry'] = (frame['e'] > frame['sqrt_q'])
    frame['short exit'] = (frame['e'] < frame['sqrt_q'])
    frame.loc[frame['short entry'], 'num units short'] = -1
    frame.loc[frame['short exit'], 'num units short'] = 0
    frame['num units short'][0] = 0
    frame['num units short'] = frame['num units short'].fillna(method='pad')

    """Note: For accurate backtesting, the hedge ratio and total investment when the
    position is opened should be kept constant until the position is closed to
    calculate the PnL."""

    frame['numUnits'] = frame['num units long'] + frame['num units short']
    frame['spread pct ch'] = (frame['spread'] - frame['spread'].shift(1)) / ((frame['x'] * abs(frame['hr'])) + frame['y'])
    frame['port rets'] = frame['spread pct ch'] * frame['numUnits'].shift(1)

    frame['cum rets'] = frame['port rets'].cumsum()
    frame['cum rets'] = frame['cum rets'] + 1

    name = "bt" + s1 + "-" + s2 + ".csv"
    frame.to_csv(name)
    ##############################################################

    try:
        sharpe = ((df1['port rets'].mean() / df1['port rets'].std()) * sqrt(252))
    except ZeroDivisionError:
        sharpe = 0.0
    ##############################################################
    start_val = 1
    end_val = df1['cum rets'].iat[-1]

    start_date = df1.iloc[0].name
    end_date = df1.iloc[-1].name
    days = (end_date - start_date).days

    CAGR = round(((float(end_val) / float(start_val)) ** (252.0 / days)) - 1, 4)

    return df1['cum rets'], sharpe, CAGR


if __name__== '__main__':

    # download and import data
    ticker_data = import_ticker_data(etf_ticker_path,
                                     start_date=start_date,
                                     end_date=end_date,
                                     time_interval=time_interval)

    # Test Kalman Filter on combination
    valid_combinations = create_valid_ticker_combs(ticker_data, min_period_yrs=min_period_yrs,
                                                   num_tickers_in_basket=num_tickers_in_basket,
                                                   max_half_life=max_half_life, min_half_life=min_half_life)

    # Filter only on valid combinations
    try:
        valid_combinations = valid_combinations.loc[valid_combinations['sample_pass'] == True]
    except KeyError:
        warnings.warn('No valid ticker combinations to process!')
        sys.exit(0)

    # For each valid combination
    valid_combinations = valid_combinations.apply(reformat_data, axis=1)

    # Test Kalman Filter code on ticker
    valid_combinations = valid_combinations.apply(apply_Kalman_filter, axis=1)


    # Estimate cumulative returns, sharpe ratio, etc without transaction costs and slippage


    # y = ticker_data['EWC']['price_df']
    # x = ticker_data['EWA']['price_df']
    #
    # etf_plot = sns.lineplot(data=pd.merge(y, x, how='inner', left_index=True, right_index=True))
    # etf_plot.figure.savefig(save_file_path('plots', 'EWC_EWA' + '.png'), format='png', dpi=1000)
    # etf_plot.figure.clf()

    # y = y.values
    # x = x.values

