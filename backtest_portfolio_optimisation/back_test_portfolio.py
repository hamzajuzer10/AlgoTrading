from backtest_portfolio_optimisation.cryptocompare_connector import fetch_all
from backtest_portfolio_optimisation.optimise_portfolio import add_all_returns, create_correlation_matrix, \
    get_portfolio_volatility, get_portfolio_return
import pandas as pd
import numpy as np
from backtest_portfolio_optimisation.utils import save_file_path

# GLOBAL
coins = ["BTC", "ETH", "ADA", "BNB", "UNI", "LTC", "XLM", "XRP", "DOT", "MATIC", "ATOM"]
coins_weights = [.0, .0, .5, .0, .0, .0, .0, .0, .0, .5, .0] # weights
periods_ago_to_fetch = [7, 14, 30, 45, 60, 75, 90]  # see also filter_history_by_date()
timescale = 'daily' # 'hourly' or 'daily'
initial_capital = 6920


def get_metrics(coins, coins_weights_list, std_deviations, correlation_matrix, cumulative_returns, timescale):

    # Define parameters
    results_array = np.zeros((3 + len(coins), 1))

    w = np.asmatrix(coins_weights_list)

    p_r = get_portfolio_return(coins, w, cumulative_returns)
    p_std = get_portfolio_volatility(w, std_deviations, correlation_matrix, timescale)

    # store results in results array
    results_array[0, 0] = p_r
    results_array[1, 0] = p_std
    # store Sharpe Ratio (return / volatility) - risk free rate element
    # excluded for simplicity
    results_array[2, 0] = results_array[0, 0] / results_array[1, 0]

    for i, w in enumerate(coins_weights_list):
        results_array[3 + i, 0] = w

    columns = ['r', 'stdev', 'sharpe'] + coins

    # convert results array to Pandas DataFrame
    results_frame = pd.DataFrame(np.transpose(results_array),
                                 columns=columns)
    # locate position of portfolio with highest Sharpe Ratio
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]

    # print("Coin weights", sess.run(coin_weights))
    print("Cumulative returns {:.2f} %".format(max_sharpe_port['r'] * 100))
    print("Annualised returns volatility {:.2f} %".format(max_sharpe_port['stdev'] * 100))
    print("Sharpe ratio {:.2f}".format(max_sharpe_port['sharpe']))


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
        coin_history = fetch_all(coins=coins, days_ago_to_fetch=days_ago_to_fetch, timescale=timescale)
        hist_length = len(coin_history[coins[0]])
        average_returns, cumulative_returns, coin_history = add_all_returns(coins, coin_history)
        correlation_matrix, pretty_matrix, std_deviations = create_correlation_matrix(coins, coin_history, hist_length)

        get_metrics(coins, coins_weights, std_deviations, correlation_matrix, cumulative_returns, timescale)

        # Generate and plot portfolio valuation by time (ignores the returns on the first day so may not be
        # the same as returns specified above
        returns = get_global_daily_valuation(coins, hist_length, coin_history, coins_weights) * initial_capital
        returns.index = pd.to_datetime(returns.index)
        returns['days'] = np.arange(len(returns))

        fig = returns.plot(x='days', y=['Cumulative return'], figsize=(15, 4)).get_figure()
        fig.savefig(
            save_file_path('plots',
                           'cumulative_returns_' + str(days_ago_to_fetch) + '_days' + '.png'),
            format='png', dpi=1000)
        fig.clf()

