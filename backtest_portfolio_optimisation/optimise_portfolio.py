import pandas as pd
import numpy as np
from backtest_portfolio_optimisation.cryptocompare_connector import fetch_all
import matplotlib.pyplot as plt


# use the following as reference: https://github.com/enigmampc/catalyst/blob/master/catalyst/examples/portfolio_optimization.py
# GLOBAL
coins = ["BTC", "ETH", "ADA", "BNB", "UNI", "LTC", "XLM", "XRP", "DOT", "ATOM"]
from_date = '2021-04-30'
to_date = '2021-05-30'
min_weight = 0.03  # min weight to invest in a crypto
max_weight = 0.4  # max weight to invest in a crypto
timescale = 'daily' # 'hourly' or 'daily'


# Calculate returns and excess returns
def add_all_returns(coins, coin_history):

    average_returns = {}
    cumulative_returns = {}
    for coin in coins:
        hist = coin_history[coin]
        hist['return'] = (hist['close'] - hist['open']) / hist['open']
        average = hist["return"].mean()
        average_returns[coin] = average
        cumulative_returns[coin] = (hist["return"] + 1).prod() - 1
        hist['excess_return'] = hist['return'] - average
        coin_history[coin] = hist

    return average_returns, cumulative_returns, coin_history


# Excess matrix
def create_correlation_matrix(coins, coin_history, hist_length):

    excess_matrix = np.zeros((hist_length, len(coins)))
    for i in range(0, hist_length):
      for idx, coin in enumerate(coins):
        excess_matrix[i][idx] = coin_history[coin].iloc[i]['excess_return']

    product_matrix = np.matmul(excess_matrix.transpose(), excess_matrix)
    var_covar_matrix = product_matrix / hist_length

    std_deviations = np.zeros((len(coins), 1))

    for idx, coin in enumerate(coins):
        std_deviations[idx][0] = np.std(coin_history[coin]['return'])

    sdev_product_matrix = np.matmul(std_deviations, std_deviations.transpose())

    correlation_matrix = var_covar_matrix / sdev_product_matrix

    # display
    pretty_matrix = pd.DataFrame(correlation_matrix).copy()
    pretty_matrix.columns = coins
    pretty_matrix.index = coins

    return correlation_matrix, pretty_matrix, std_deviations


def get_portfolio_volatility(w, std_deviations, correlation_matrix, timescale):

    weighted_std_devs = np.multiply(np.transpose(w), std_deviations)

    product_1 = np.transpose(weighted_std_devs)
    product_2 = np.matmul(product_1, correlation_matrix)
    variance = np.matmul(product_2, weighted_std_devs)

    # Annualize volatility
    if timescale == 'daily':
        vol_mult = np.sqrt(365)
    elif timescale == 'hourly':
        vol_mult = np.sqrt(365 * 24)

    return np.sum(np.sqrt(variance)) * vol_mult


def get_portfolio_return(coins, w, cumulative_returns):

    # Compute cumulative returns
    m = np.full((len(coins), 1), 0.0)  # same as coin_weights
    for coin_idx in range(0, len(coins)):
        m[coin_idx] = cumulative_returns[coins[coin_idx]]

    # return weighted return
    return sum(np.dot(w, m))


# Optimize weights to maximize return/risk (Sharpe ratio)
def maximize_sharpe_ratio(coins, std_deviations, cumulative_returns, correlation_matrix, min_weight, max_weight, timescale):

    # Define portfolio optimization parameters
    n_portfolios = 100000
    results_array = np.zeros((3 + len(coins), n_portfolios))
    for p in range(n_portfolios):
        weights = np.random.random(len(coins))
        weights /= np.sum(weights)
        w = np.asmatrix(weights)

        p_r = get_portfolio_return(coins, w, cumulative_returns)
        p_std = get_portfolio_volatility(w, std_deviations, correlation_matrix, timescale)

        # store results in results array
        results_array[0, p] = p_r
        results_array[1, p] = p_std
        # store Sharpe Ratio (return / volatility) - risk free rate element
        # excluded for simplicity
        results_array[2, p] = results_array[0, p] / results_array[1, p]

        for i, w in enumerate(weights):
            results_array[3 + i, p] = w

    columns = ['r', 'stdev', 'sharpe'] + coins

    # convert results array to Pandas DataFrame
    results_frame = pd.DataFrame(np.transpose(results_array),
                                 columns=columns)
    # locate position of portfolio with highest Sharpe Ratio
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
    # locate positon of portfolio with minimum standard deviation
    # min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]

    # create scatter plot coloured by Sharpe Ratio
    plt.scatter(results_frame.stdev,
                results_frame.r,
                c=results_frame.sharpe,
                cmap='RdYlGn')
    plt.xlabel('Volatility')
    plt.ylabel('Returns')
    plt.colorbar()

    # plot blue circle to highlight position of portfolio
    # with highest Sharpe Ratio
    plt.scatter(max_sharpe_port[1],
                max_sharpe_port[0],
                marker='o',
                color='b',
                s=200)

    plt.show()
    print("Cumulative returns {:.2f} %".format(max_sharpe_port['r']*100))
    print("Annualised returns volatility {:.2f} %".format(max_sharpe_port['stdev']*100))
    print("Sharpe ratio {:.2f}".format(max_sharpe_port['sharpe']))
    return max_sharpe_port

if __name__ == '__main__':

    coin_history = fetch_all(coins=coins, start_date=from_date, end_date=to_date, timescale=timescale)

    hist_length = len(coin_history[coins[0]])
    average_returns, cumulative_returns, coin_history = add_all_returns(coins, coin_history)
    correlation_matrix, pretty_matrix, std_deviations = create_correlation_matrix(coins, coin_history, hist_length)

    weights = maximize_sharpe_ratio(coins, std_deviations, cumulative_returns, correlation_matrix, min_weight, max_weight, timescale)

    pretty_weights = weights[coins]
    print(pretty_weights)
