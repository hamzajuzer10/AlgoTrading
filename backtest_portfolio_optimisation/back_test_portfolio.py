from backtest_portfolio_optimisation.optimise_portfolio import fetch_all, add_all_returns, create_correlation_matrix, \
    get_portfolio_volatility, get_portfolio_return
import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt


# GLOBAL
coins = ["BTC", "ETH", "ADA", "BNB", "UNI", "LTC", "XLM", "XRP"] # coins
coins_weights = [.04948837, .27709115, .04949638, .26242672, .04952015, .21300072, .04948826, .04948826] # weights
periods_ago_to_fetch = [30, 45, 60, 75, 90]  # see also filter_history_by_date()
initial_capital = 6200


def get_metrics(coins, coins_weights_list, std_deviations, correlation_matrix, cumulative_returns):

    coin_weights = tf.constant(coins_weights_list, shape=(8,1), dtype=tf.float64)  # our variables

    portfolio_volatility = get_portfolio_volatility(coin_weights, std_deviations, correlation_matrix)

    portfolio_return = get_portfolio_return(coins, coin_weights, cumulative_returns)

    # 3) Return / Risk
    sharpe_ratio = tf.divide(portfolio_return, portfolio_volatility)

    with tf.Session() as sess:

        # print("Coin weights", sess.run(coin_weights))
        print("Daily returns volatility {:.2f} %".format(sess.run(tf.reduce_sum(portfolio_volatility))))
        print("Total portfolio return {:.2f} %".format(sess.run(portfolio_return) * 100))
        print("Sharpe ratio", sess.run(tf.reduce_sum(sharpe_ratio)))


def get_cumulative_ratios_by_coin(coins, hist_length, coin_history):
    result = np.zeros((hist_length, len(coins)))
    result[0] = np.full(len(coins), 1)

    for i in range(1, hist_length):
        for idx, coin in enumerate(coins):
            result[i][idx] = result[i - 1][idx] * (coin_history[coin].iloc[i]['return'] + 1)

    return result

def get_global_daily_valuation(coins, hist_length, coin_history, weights):
    cumulative_ratios = get_cumulative_ratios_by_coin(coins, hist_length, coin_history)
    daily_cumulative_ratios = np.zeros(hist_length)
    for i in range(hist_length):
        daily_cumulative_ratios[i] = np.matmul(cumulative_ratios[i], np.array(weights))

    return pd.DataFrame(daily_cumulative_ratios, columns=["Cumulative return"], index=coin_history[coins[0]].index)


if __name__ == '__main__':

    for days_ago_to_fetch in periods_ago_to_fetch:
        print('Time period: {} days'.format(days_ago_to_fetch))
        coin_history = fetch_all(coins, days_ago_to_fetch)
        hist_length = len(coin_history[coins[0]])
        average_returns, cumulative_returns, coin_history = add_all_returns(coins, coin_history)
        correlation_matrix, pretty_matrix, std_deviations = create_correlation_matrix(coin_history, hist_length)

        get_metrics(coins, coins_weights, std_deviations, correlation_matrix, cumulative_returns)

    # Generate and plot portfolio valuation by time (ignores the returns on the first day so may not be
    # the same as returns specified above
    returns = get_global_daily_valuation(coins, hist_length, coin_history, coins_weights) * initial_capital
    returns.index = pd.to_datetime(returns.index)
    returns['days'] = np.arange(len(returns))
    returns.plot(x='days', y=['Cumulative return'], figsize=(15, 4))

    plt.xticks(rotation=15)
    plt.title('Cumulative returns')
    plt.show()

