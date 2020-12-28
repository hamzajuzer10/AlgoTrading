import pandas as pd
from backtest_pairs.yfinance_connector import import_ticker_data, load_ticker_data_json
from backtest_pairs.process_pairs import build_price_df, group_timeZone, create_combinations, filter_high_liquidity_tickers
from itertools import combinations
from io import StringIO
import sys
import warnings
from collections import defaultdict
import numpy as np

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

json_path = "C:\\Users\\hamzajuzer\\Documents\\Algorithmic Trading\AlgoTradingv1\\backtest_pairs\\data\\etf_tickers.json"
csv_path = "C:\\Users\\hamzajuzer\\Documents\\Algorithmic Trading\AlgoTradingv1\\backtest_pairs\\pyspark\\etf_tickers.csv"
num_tickers_in_basket = 2 # max 12
start_date = '2006-04-26'
end_date = '2012-04-09'
time_interval = 'daily'
time_zones = [-14400]


def create_ticker_combs_csv(ticker_data, num_tickers_in_basket: int,
                           formatted_file_path, time_zones=None):

    # only consider tickers with sufficient liquidity
    ticker_data = filter_high_liquidity_tickers(ticker_data)

    # create a price df, timezone and currency for each ticker
    ticker_data = build_price_df(ticker_data)

    # create num_tickers_in_basket combinations of ticker data, grouping by timeZone
    time_zone_ticker_groups = group_timeZone(ticker_data)

    # Filter on timezones for faster processing
    if time_zones:
        time_zone_ticker_groups = dict(
            (k, time_zone_ticker_groups[k]) for k in time_zones if k in time_zone_ticker_groups)

    list_comb = create_combinations(time_zone_ticker_groups, num_tickers_in_basket)

    if len(list_comb) == 0:
        warnings.warn('No combination of tickers generated!')

    # For each combination, check the minimum start dates for all tickers in basket
    for index, i in enumerate(list_comb):
        # johansen test
        # merge all dataframes together
        print('Processing combination {a}/{b}: {c}'.format(a=index, b=len(list_comb), c=i))
        merged_prices_comb_df = pd.DataFrame()
        for t in range(len(i)):

            j = i[t]
            col_name = 'x_'+str(t)
            ticker_data[j]['price_df'].rename(columns={ticker_data[j]['price_df'].columns[0]: col_name}, inplace=True)
            merged_prices_comb_df = pd.merge(merged_prices_comb_df, ticker_data[j]['price_df'], left_index=True,
                                             right_index=True, how='outer')

        # add a col for combination
        merged_prices_comb_df['combination'] = '_'.join(i)

        # drop all na rows
        merged_prices_comb_df.dropna(inplace=True)

        if index == 0:
            merged_prices_comb_df.to_csv(formatted_file_path, mode='w', index=True)
        else:
            merged_prices_comb_df.to_csv(formatted_file_path, mode='a', index=True, header=False)


if __name__== '__main__':

    # download and import data
    print('Importing ticker data')
    # ticker_data = import_ticker_data(tickers=['EWA', 'EWC', 'DIA', 'IYT'],
    #                                  start_date=start_date,
    #                                  end_date=end_date,
    #                                  time_interval=time_interval)
    ticker_data = load_ticker_data_json(json_path)

    # calculating valid ticker combinations
    print('Reformatting ticker combinations and saving into csv')
    create_ticker_combs_csv(ticker_data, num_tickers_in_basket=num_tickers_in_basket,
                            formatted_file_path=csv_path, time_zones=time_zones)