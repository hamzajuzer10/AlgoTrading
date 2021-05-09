import json
import requests
import pandas as pd
import numpy as np
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import matplotlib.pyplot as plt

# GLOBAL
coins = ["BTC", "ETH", "ADA", "BNB", "UNI", "LTC", "XLM", "XRP"]
days_ago_to_fetch = 30  # see also filter_history_by_date()
min_weight = 0.05  # min weight to invest in a crypto
max_weight = 0.4  # max weight to invest in a crypto


def fetch_all(coins, days_ago_to_fetch):

    coin_history = {}
    for coin in coins:
        coin_history[coin] = fetch_history(coin, days_ago_to_fetch)

    return coin_history


def fetch_history(coin, days_ago_to_fetch):
    endpoint_url = "https://min-api.cryptocompare.com/data/histoday?fsym={}&tsym=USD&limit={:d}".format(coin,
                                                                                                        days_ago_to_fetch)
    res = requests.get(endpoint_url)
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = index_history(hist)
    hist = filter_history_by_date(hist)
    return hist


def index_history(hist):
    # index by date so we can easily filter by a given timeframe
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    return hist


def filter_history_by_date(hist):
    result = hist[hist.index.year >= 2017]
    # result = result[result.index.day == 1] # every first of month, etc.
    return result


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
def create_correlation_matrix(coin_history, hist_length):

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


def get_portfolio_volatility(coin_weights, std_deviations, correlation_matrix):
    weighted_std_devs = tf.multiply(coin_weights, std_deviations)

    product_1 = tf.transpose(weighted_std_devs)
    product_2 = tf.matmul(product_1, correlation_matrix)
    variance = tf.matmul(product_2, weighted_std_devs)

    volatility = tf.sqrt(variance)
    return tf.reduce_sum(volatility)


def get_portfolio_return(coins, coin_weights, cumulative_returns):
    returns = np.full((len(coins), 1), 0.0)  # same as coin_weights
    for coin_idx in range(0, len(coins)):
        returns[coin_idx] = cumulative_returns[coins[coin_idx]]

    p_return = tf.multiply(coin_weights, returns)
    return tf.reduce_sum(p_return)


def ensure_constraints_op(coin_weights, min_weight, max_weight):
    # Constraints
    # all values positive, with unity sum
    # weights_sum = tf.reduce_sum(coin_weights)
    # constraints_op = coin_weights.assign(tf.divide(tf.abs(coin_weights), tf.abs(weights_sum)))

    lower_than_x = tf.greater(np.float64(min_weight), coin_weights)
    zero_minimum_op = coin_weights.assign(tf.where(lower_than_x, tf.ones_like(coin_weights) * min_weight, coin_weights))

    greater_than_x = tf.greater(coin_weights, np.float64(max_weight))
    unity_max_op = coin_weights.assign(tf.where(greater_than_x, tf.ones_like(coin_weights) * max_weight, coin_weights))

    result_sum = tf.reduce_sum(coin_weights)
    unity_sum_op = coin_weights.assign(tf.divide(coin_weights, result_sum))

    constraints_op = tf.group(zero_minimum_op, unity_max_op, unity_sum_op)

    return constraints_op


# Optimize weights to maximize return/risk (Sharpe ratio)
def maximize_sharpe_ratio(coins, std_deviations, cumulative_returns, min_weight, max_weight):
    # Define the model

    coin_weights = tf.Variable(tf.random_uniform((len(coins), 1), dtype=tf.float64))  # our variables

    portfolio_volatility = get_portfolio_volatility(coin_weights, std_deviations, correlation_matrix)

    portfolio_return = get_portfolio_return(coins, coin_weights, cumulative_returns)

    # 3) Return / Risk
    sharpe_ratio = tf.divide(portfolio_return, portfolio_volatility)

    constraints_op = ensure_constraints_op(coin_weights, min_weight, max_weight)

    # Run
    #learning_rate = 0.0001
    learning_rate = 0.000005
    steps = 50000

    # Training using Gradient Descent to minimize cost

#    optimize_op = tf.train.GradientDescentOptimizer(learning_rate, use_locking=True).minimize(tf.negative(sharpe_ratio))
    # optimize_op = tf.train.AdamOptimizer(learning_rate, use_locking=True).minimize(tf.divide(1, sharpe_ratio))

    # 2# optimize_op = tf.train.AdamOptimizer(learning_rate, use_locking=True).minimize(tf.negative(sharpe_ratio))
    optimize_op = tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(tf.divide(1, sharpe_ratio))
    #optimize_op = tf.train.AdagradOptimizer(learning_rate=0.01, initial_accumulator_value=0.1, use_locking=False).minimize(tf.divide(1, sharpe_ratio))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        ratios = np.zeros(steps)
        returns = np.zeros(steps)
        sess.run(init)
        for i in range(steps):
            sess.run(optimize_op)
            sess.run(constraints_op)
            ratios[i] = sess.run(sharpe_ratio)
            returns[i] = sess.run(portfolio_return) * 100
            if i % 2000 == 0:
                sess.run(constraints_op)
                print("[round {:d}]".format(i))
                # print("Coin weights", sess.run(coin_weights))
                print("Daily returns volatility {:.2f} %".format(sess.run(portfolio_volatility)))
                print("Total portfolio return {:.2f} %".format(sess.run(portfolio_return) * 100))
                print("Sharpe ratio", sess.run(sharpe_ratio))
                print("")

        sess.run(constraints_op)
        # print("Coin weights", sess.run(coin_weights))
        print("Daily returns volatility {:.2f} %".format(sess.run(portfolio_volatility)))
        print("Total portfolio return {:.2f} %".format(sess.run(portfolio_return) * 100))
        print("Sharpe ratio", sess.run(sharpe_ratio))
        return sess.run(coin_weights)


if __name__ == '__main__':

    coin_history = fetch_all(coins, days_ago_to_fetch)
    hist_length = len(coin_history[coins[0]])
    average_returns, cumulative_returns, coin_history = add_all_returns(coins, coin_history)
    correlation_matrix, pretty_matrix, std_deviations = create_correlation_matrix(coin_history, hist_length)

    weights = maximize_sharpe_ratio(coins, std_deviations, cumulative_returns, min_weight, max_weight)

    pretty_weights = pd.DataFrame(weights * 100, index=coins, columns=["Weight %"])
    print(pretty_weights)
