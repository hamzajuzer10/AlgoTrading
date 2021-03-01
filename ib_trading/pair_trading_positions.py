# Kalman filter to estimate dynamic portfolio weights
from backtest_pairs.kalman_filter import MyKalmanPython
from backtest_pairs.yfinance_connector import import_ticker_data
from backtest_pairs.process_pairs import build_price_df
from datetime import datetime, date
from itertools import combinations
import warnings
from backtest_pairs.utils import save_file_path
import numpy as np
import pandas as pd
from math import sqrt, floor


def get_optimal_sizing(prices: np.object, betas: np.object, cash: float, risk: float, round_position=True):
    '''Get the optimal unit size'''

    trade_cash = cash * risk
    unit_price = prices.dot(abs(betas))  # assumes prices are all positive and not trading shorts on margin (i.e. hold full amt)
    size = trade_cash / unit_price

    if round_position:
        size = int(floor(size))

    return size


def create_initial_csv(csv_path: str,
                       tickers: list,
                       start_date: str, end_date: str,
                       time_interval: str, use_close_prices: bool,
                       cash: float,
                       entry_sqrt_q_multiplier=1,
                       exit_sqrt_q_multiplier=0):
    """Create the initial csv files containing prices"""

    # benchmark data
    print('Importing ticker data...')
    ticker_data = import_ticker_data(tickers=tickers,
                                     start_date=start_date,
                                     end_date=end_date,
                                     time_interval='daily')

    print('Building price_df...')
    ticker_data = build_price_df(ticker_data, time_interval=time_interval, use_close_prices=use_close_prices)

    valid_combinations = []
    comb = combinations(ticker_data.keys(), len(ticker_data.keys()))
    valid_combinations.extend(list(comb))

    if len(valid_combinations) == 0:
        warnings.warn('No combination of tickers generated!')


    for index, i in enumerate(valid_combinations):

        # merge all dataframes together
        merged_prices_comb_df = pd.DataFrame()
        for j in i:
            merged_prices_comb_df = pd.merge(merged_prices_comb_df, ticker_data[j]['price_df'], left_index=True,

                                             right_index=True, how='outer')

        merged_prices_comb_df.dropna(inplace=True)

    merged_prices_comb_df['rank'] = np.arange(merged_prices_comb_df.shape[0])
    merged_prices_comb_df['portfolio_position'] = None # will be either 'long' or 'short'
    merged_prices_comb_df['position_size'] = 0 # size of position in units
    merged_prices_comb_df['entry_sqrt_q_multiplier'] = entry_sqrt_q_multiplier
    merged_prices_comb_df['exit_sqrt_q_multiplier'] = exit_sqrt_q_multiplier
    merged_prices_comb_df['commission'] = None
    merged_prices_comb_df['short_interest'] = None
    merged_prices_comb_df['comments'] = None

    # add field for positions of y and x ticker (units)
    for ticker in tickers:
        ticker_pos = ticker + '_position'
        merged_prices_comb_df[ticker_pos] = 0

    # add field for cash
    merged_prices_comb_df['cash'] = cash

    # save price df to file
    print('Saving to csv file...')

    try:
        merged_prices_comb_df.to_csv(csv_path, mode='x')

    except FileExistsError:
        raise FileExistsError("Error: File exists, please manually delete the file before re-running...!")

    return None


def read_pair_csvs(csv_path: str,
                   observation_date: str, # '2019-12-31'
                   y_ticker: str,
                   observed_y_ticker_price: float,
                   x_tickers_list: list,
                   observed_x_ticker_price: list):

    # Read csv file with data
    print('Reading csv data...')
    merged_prices_comb_df = pd.read_csv(csv_path)

    # check that x and y tickers exist as cols in the dataframe
    ticker_list = x_tickers_list + [y_ticker]
    if not set(ticker_list).issubset(merged_prices_comb_df.columns):

        print("The y and x tickers specified don't exist as columns in the csv file")
        raise ValueError

    # warn if the the observation date is not today
    obs_date_datetime = datetime.strptime(observation_date, '%Y-%m-%d')
    date_of_today = date.today()
    if obs_date_datetime.date() != date_of_today:
        warnings.warn("Observed date is not today's date...!")

    # add the observed prices at the end
    observed_dict = {'formatted_date': datetime.strptime(observation_date, '%Y-%m-%d'),
                      y_ticker: observed_y_ticker_price}

    for i in range(len(x_tickers_list)):
        observed_dict[x_tickers_list[i]] = observed_x_ticker_price[i]

    merged_prices_comb_df = merged_prices_comb_df.append(observed_dict, ignore_index=True)

    merged_prices_comb_df = merged_prices_comb_df.set_index('formatted_date')
    merged_prices_comb_df.index = pd.to_datetime(merged_prices_comb_df.index, format='%Y-%m-%d')

    # sort by date
    merged_prices_comb_df.sort_values(by=['formatted_date'], inplace=True)

    # get initial mean which is zero
    initial_mean = [0]*(1+len(x_tickers_list))

    # generate pair positions for the observed prices
    kf_output = generate_pair_positions(y_series=merged_prices_comb_df[y_ticker],
                                        x_df=merged_prices_comb_df[x_tickers_list],
                                        init_state_mean=initial_mean,
                                        dim=(1+len(x_tickers_list)),
                                        y_ticker=y_ticker,
                                        x_ticker=x_tickers_list,
                                        price_averaging=False)

    # merge with merged price comb
    kf_output.drop(ticker_list, axis=1, inplace=True)
    merged_prices_comb_df = pd.merge(merged_prices_comb_df, kf_output, left_index=True, right_index=True, how="inner")
    merged_prices_comb_df.sort_values(by=['formatted_date'])

    # generate positions from outputs
    merged_prices_comb_df_tail = merged_prices_comb_df.tail(2)
    prev_position_df = merged_prices_comb_df_tail.head(1)
    current_position_df = merged_prices_comb_df_tail.tail(1)

    # get previous position, previous cash and entry/ exit multipliers
    prev_position = prev_position_df.iloc[0]['portfolio_position']
    entry_sqrt_q_multiplier = prev_position_df.iloc[0]['entry_sqrt_q_multiplier']
    exit_sqrt_q_multiplier = prev_position_df.iloc[0]['exit_sqrt_q_multiplier']

    # get current e and sqrt q
    e = current_position_df.iloc[0]['e']
    sqrt_q = current_position_df.iloc[0]['sqrt_q']


    if prev_position == 'long' and e >= -(sqrt_q * exit_sqrt_q_multiplier):

        # close your position
        print('Close your existing position')
        prev_position = np.nan

    elif prev_position == 'short' and e <= (sqrt_q * exit_sqrt_q_multiplier):

        # close your position
        print('Close your existing position')
        prev_position = np.nan

    elif prev_position == 'long' or prev_position == 'short':

        # re-balance your position
        print('Re-balance your position')

        # long or short x (only need to rebalance x)
        for x_ticker in x_tickers_list:

            # get previous position
            x_pos = x_ticker + '_position'
            prev_x_position = prev_position_df.iloc[0][x_pos]

            # get position size
            position_size = prev_position_df.iloc[0]['position_size']

            # calculate new position
            beta_col = x_ticker + '_beta'
            beta = current_position_df.iloc[0][beta_col]

            # determine if we need to long or short
            if prev_position == 'long':
                # short
                new_x_position = -position_size*beta
            else:
                # long
                new_x_position = position_size * beta

            print('Target position for {ticker} is {size}'.format(ticker=x_ticker, size=new_x_position))

            # calculate difference
            diff = new_x_position-prev_x_position
            print('Order {size_diff} for {ticker}'.format(ticker=x_ticker, size_diff=diff))

    if pd.isnull(prev_position):

        # Calculate optimal size
        prices_mat = np.zeros((1 + len(x_tickers_list)))
        betas_mat = np.ones((1 + len(x_tickers_list)))

        prices_mat[0] = current_position_df.iloc[0][y_ticker]

        if e < -(sqrt_q * entry_sqrt_q_multiplier):
            betas_mat[0] = 1 # long y

        elif e > (sqrt_q * entry_sqrt_q_multiplier):
            betas_mat[0] = -1  # short y

        else:
            return None

        for i in range(len(x_tickers_list)):
            x_ticker = x_tickers_list[i]
            prices_mat[i + 1] = current_position_df.iloc[0][x_ticker]

            beta_col = x_ticker + '_beta'
            betas_mat[i + 1] = -1 * betas_mat[0] * current_position_df.iloc[0][beta_col]

        # prompt for cash
        print("Portfolio entry conditions triggered...")
        cash = input("Enter your investment size (USD): \n")
        cash = float(cash)

        opt_size = get_optimal_sizing(prices=prices_mat,
                                      betas=betas_mat,
                                      cash=cash, risk=1)

        print('{y_ticker} unit size: {size}'.format(y_ticker=y_ticker, size=opt_size*betas_mat[0]))

        for i in range(len(x_tickers_list)):
            x_ticker = x_tickers_list[i]
            x_ticker_size = betas_mat[i + 1] * opt_size
            print('{x_ticker} unit size: {size}'.format(x_ticker=x_ticker, size=x_ticker_size))

    return None


def save_pair_csvs(csv_path: str,
                   observation_date: str, # '2019-12-31'
                   y_ticker: str,
                   purchased_y_ticker_price: float,
                   purchased_y_ticker_units: float,
                   x_tickers_list: list,
                   purchased_x_ticker_price: list,
                   purchased_x_ticker_units: list,
                   cash: float,
                   commission=None,
                   short_interest=0,
                   entry_sqrt_q_multiplier=1,
                   exit_sqrt_q_multiplier=0,
                   comments=None):

    # Read csv file with data
    print('Reading csv data...')
    merged_prices_comb_df = pd.read_csv(csv_path)

    # check that x and y tickers exist as cols in the dataframe
    ticker_list = x_tickers_list + [y_ticker]
    if not set(ticker_list).issubset(merged_prices_comb_df.columns):
        print("The y and x tickers specified don't exist as columns in the csv file")
        raise ValueError

    # warn if the the observation date is not today
    obs_date_datetime = datetime.strptime(observation_date, '%Y-%m-%d')
    date_of_today = date.today()
    if obs_date_datetime.date() != date_of_today:
        warnings.warn("Observed date is not today's date...!")

    # warn if commissions and short interest is +ve
    if commission > 0 or short_interest > 0:
        warnings.warn("Commissions or short interest is +ve where it should be -ve...!")

    if (purchased_y_ticker_units > 0) and (sum(purchased_x_ticker_units) < 0):
        portfolio_position = 'long'
    elif (purchased_y_ticker_units < 0) and (sum(purchased_x_ticker_units) > 0):
        portfolio_position = 'short'
    elif (purchased_y_ticker_units == 0) and (sum(purchased_x_ticker_units) == 0):
        portfolio_position = None
    else:
        raise ValueError("Incorrect portfolio positions specified!")

    # add the observed prices at the end
    y_ticker_pos = y_ticker + '_position'

    observed_dict = {'formatted_date': datetime.strptime(observation_date, '%Y-%m-%d'),
                     'portfolio_position': portfolio_position,
                     'position_size': abs(purchased_y_ticker_units),
                     y_ticker: purchased_y_ticker_price,
                     y_ticker_pos: purchased_y_ticker_units,
                     'entry_sqrt_q_multiplier': entry_sqrt_q_multiplier,
                     'exit_sqrt_q_multiplier': exit_sqrt_q_multiplier,
                     'cash': cash,
                     'commission': commission,
                     'short_interest': short_interest,
                     'comments': comments}

    for i in range(len(x_tickers_list)):
        observed_dict[x_tickers_list[i]] = purchased_x_ticker_price[i]

        x_ticker_pos = x_tickers_list[i] + '_position'
        observed_dict[x_ticker_pos] = purchased_x_ticker_units[i]

    merged_prices_comb_df = merged_prices_comb_df.append(observed_dict, ignore_index=True)
    merged_prices_comb_df = merged_prices_comb_df.set_index('formatted_date')
    merged_prices_comb_df.index = pd.to_datetime(merged_prices_comb_df.index, format='%Y-%m-%d %H:%M:%S')

    # sort by date
    merged_prices_comb_df.sort_values(by=['formatted_date'], inplace=True)

    # fill rank col
    merged_prices_comb_df['rank'] = np.arange(merged_prices_comb_df.shape[0])

    # save to csv
    print('Saving to csv file...')

    # prompt for validation
    print("\n".join("{}\t{}".format(k, v) for k, v in observed_dict.items()))
    val = input("Ensure details are correct (y/n): \n")

    if val.lower() == 'y':
        merged_prices_comb_df.to_csv(csv_path)

    else:
        print("Re-run the function!")

    return None


def generate_pair_positions(y_series: pd.Series, x_df: pd.DataFrame,
                            init_state_mean, dim: int, y_ticker: str, x_ticker: list,
                            price_averaging=False):

    """Generate pair positions for every new price observation"""

    # Add y observed price and x_observed prices to the end of the price series data frame


    x_mat = np.ones((y_series.shape[0], dim))
    for i in range(len(x_ticker)):
        x_ticker_ = x_ticker[i]
        x_mat[:, i] = (x_df[x_ticker_])

    kf = MyKalmanPython(init_state_mean=init_state_mean,
                        Ve=0.001, delta=0.0001, n_dim_state=dim, use_kalman_price_averaging=price_averaging)

    for idx in range(0, y_series.shape[0]):

        obs_mat = np.asarray([x_mat[idx, :]])
        kf.update(y_mat=y_series.iloc[idx],
                  x_mat=obs_mat)


    beta_cols = x_ticker
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

    return kf_soln


if __name__ == '__main__':

    # Create an initial csv file for # KFYP, TMF combination
    # create_initial_csv(csv_path="ib_trading/shadow_trading_data/y_EIS_x_PLTM.csv",
    #                    tickers=['EIS', 'PLTM'],
    #                    start_date='2020-09-01',
    #                    end_date='2021-01-26',
    #                    time_interval='weekly',
    #                    use_close_prices=True,
    #                    cash=10000)

    # read csv
    read_pair_csvs(csv_path="ib_trading/shadow_trading_data/y_EIS_x_PLTM.csv",
                   observation_date='2021-04-01',
                   y_ticker='EIS',
                   observed_y_ticker_price=65.66,
                   x_tickers_list=['PLTM'],
                   observed_x_ticker_price=[11.70])

    # write csv
    # save_pair_csvs(csv_path="ib_trading/shadow_trading_data/y_EIS_x_PLTM.csv",
    #                observation_date='2021-03-01',  # '2019-12-31'
    #                y_ticker='EIS',
    #                purchased_y_ticker_price=65.71,
    #                purchased_y_ticker_units=0,
    #                x_tickers_list=['PLTM'],
    #                purchased_x_ticker_price=[11.72],
    #                purchased_x_ticker_units=[0],
    #                cash=999695,
    #                commission=-1.03,
    #                comments="No EIS available to short - otherwise, advised to short EIS long PLTM!")

