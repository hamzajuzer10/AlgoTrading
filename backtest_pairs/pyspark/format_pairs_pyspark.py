import pandas as pd
from backtest_pairs.yfinance_connector import import_ticker_data, load_ticker_data_json
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


class StdoutRedirection:
    """Standard output redirection context manager"""

    def __init__(self, path, write_mode="w"):
        self._result = StringIO()
        self._path = path
        self._write_mode = write_mode

    def __enter__(self):
        sys.stdout = self._result
        # sys.stdout = open(self._path, mode=self._write_mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # sys.stdout.close()
        sys.stdout = sys.__stdout__

    def save_to_txt_file(self):

        with open(self._path, self._write_mode) as out_file:
            out_file.write(self._result.getvalue())


def filter_high_liquidity_tickers(data: dict, n_days = 90, min_last_n_day_vol: float = 10000):

    # filter tickers with average 30 days vol > min
    for ticker in data:
        try:
            vol_df = pd.DataFrame(data[ticker]['prices'])
            vol_df = vol_df.drop('date', axis=1).set_index('formatted_date')
            vol_df.index = pd.to_datetime(vol_df.index, format='%Y-%m-%d %H:%M:%S')
            vol_df = vol_df[['volume']]
            vol_mean = np.mean(vol_df.tail(n_days)).item()
            if vol_mean < min_last_n_day_vol:
                print('Ticker {b} volume traded is smaller than min, {a}. Disregarding ticker!'.format(min_last_n_day_vol, ticker))
                del data[ticker]

        except KeyError:
            warnings.warn('Volume data not available for {}!'.format(ticker))
            del data[ticker]

        return data


def build_price_df(data: dict):

    output_dict = {}

    # if prices, currency, instrumentType and timeZone is a key
    for ticker in data:
        try:
            price_df = pd.DataFrame(data[ticker]['prices'])
            price_df = price_df.drop('date', axis=1).set_index('formatted_date')
            price_df.index = pd.to_datetime(price_df.index, format='%Y-%m-%d %H:%M:%S')
            price_df = price_df[['adjclose']]
            price_df.rename(columns={'adjclose': ticker}, inplace=True)
            output_dict[ticker] = {}
            output_dict[ticker]['price_df'] = price_df
            output_dict[ticker]['currency'] = data[ticker]['currency']
            output_dict[ticker]['instrumentType'] = data[ticker]['instrumentType']
            output_dict[ticker]['timeZone_gmtOffset'] = data[ticker]['timeZone']['gmtOffset']

        except KeyError:
            warnings.warn('Price, currency, instrumentType or timeZone data not available for {}!'.format(ticker))

    return output_dict


def group_timeZone(data: dict):

    # group tickers with the same timezone
    output = defaultdict(list)

    for ticker, value in data.items():

        output[value['timeZone_gmtOffset']].append(ticker)

    return output


def create_combinations(data: dict, n_tickers_in_basket: int):

    # create combinations
    valid_combinations = []
    for key, value in data.items():
        comb = combinations(value, n_tickers_in_basket)
        valid_combinations.extend(list(comb))

    return valid_combinations


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