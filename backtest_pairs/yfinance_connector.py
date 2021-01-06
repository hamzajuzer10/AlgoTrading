import pandas as pd
from yahoofinancials import YahooFinancials
import warnings
import sys
import os
import json
from threading import Thread
import functools

etf_ticker_path = "backtest_pairs\\data\\etf_tickers_12_2020.csv"
json_path = "backtest_pairs\\data\\etf_tickers_12_2020.json"
no_data_json_path = "backtest_pairs\\data\\etf_tickers_no_data_12_2020.json"
start_date = '2018-06-01'
end_date = '2020-12-01'
time_interval = 'weekly'


def timeout(seconds_before_timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, seconds_before_timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(seconds_before_timeout)
            except Exception as e:
                print('error starting thread')
                raise e
            ret = res[0]
            # if isinstance(ret, BaseException):
                # raise ret
            return ret
        return wrapper
    return deco


def processJSON(json_path, ticker_data=None):

    if os.path.exists(json_path):
        with open(json_path, mode='r', encoding='utf-8') as f:
            try:
                feeds = json.load(f)
            except Exception as e:
                warnings.warn("%s on json.load()!" % e)

    else:
        feeds={}

    if not ticker_data:
        return feeds

    if isinstance(ticker_data, dict):
        with open(json_path, mode='w+', encoding='utf-8') as feedsjson:
            feeds = {**feeds, **ticker_data}
            json.dump(feeds, feedsjson, ensure_ascii=False, indent=4)


@timeout(5)
def import_yahoo_data(ticker,
                      start_date,
                      end_date,
                      time_interval):

    yahoo_financials = YahooFinancials(ticker)

    # download data
    data = yahoo_financials.get_historical_price_data(start_date=start_date,
                                                      end_date=end_date,
                                                      time_interval=time_interval)

    return data


def import_ticker_data(start_date: str, end_date: str, time_interval: str,
                       ticker_file_path: str = None, tickers: list = None, save_json_path: str = None,
                       no_data_json_path: str = None, save_data_only=False):

    # Yahoo Finance returns Xccy in Exchange Xccy i.e NYSE in USD, LSE in GBx
    # read csv file with ticker data
    if ticker_file_path:
        ticker_df = pd.read_csv(ticker_file_path)
        ticker_list = ticker_df['Symbol'].tolist()
    elif tickers:
        ticker_list = tickers
    else:
        warnings.warn('No file path or list of tickers provided')
        sys.exit(0)

    # save ticker data in json in current directory

    if save_json_path:
        # read json file if exists
        feeds = processJSON(save_json_path)
        no_data_feeds = processJSON(no_data_json_path)

        downloaded_tickers = feeds.keys()
        del feeds # save memory
        print('No. of tickers downloaded: {}'.format(len(downloaded_tickers)))

        n_data_tickers = no_data_feeds.keys()
        del no_data_feeds # save memory
        print('No. of tickers without yahoo data: {}'.format(len(n_data_tickers)))

        ticker_list = [x for x in ticker_list if x not in downloaded_tickers]
        ticker_list = [x for x in ticker_list if x not in n_data_tickers]

        print('No. of tickers to download: {}'.format(len(ticker_list)))

    upload_data = {}
    all_data = {}
    no_data = {}
    i = 0
    for ticker in ticker_list:

        i += 1

        if no_data_json_path:

            if (len(no_data.keys()) > 0 and len(no_data.keys()) % 250 == 0) or (i == len(ticker_list)):
                # update no data json file every n keys or at the end
                print('Saving no data tickers to JSON file...')
                processJSON(no_data_json_path, ticker_data=no_data)

        if save_json_path:

            if (len(upload_data.keys()) > 0 and len(upload_data.keys()) % 250 == 0) or (i == len(ticker_list)):
                # update json file every n keys or at the end
                print('Saving ticker data to JSON file...')
                processJSON(save_json_path, ticker_data=upload_data)
                upload_data = {}

        print('Downloading ticker {a}/{b}: {c}'.format(a=i, b=len(ticker_list), c=ticker))

        # download data
        data = import_yahoo_data(ticker=ticker,
                                 start_date=start_date,
                                 end_date=end_date,
                                 time_interval=time_interval)

        if isinstance(data, Exception):
            warnings.warn('Timeout error - data not available for {}!'.format(ticker))
            no_data[ticker] = "No data"

            continue

        if save_json_path:
            upload_data = {**upload_data, **data}

        if not save_data_only:
            all_data = {**all_data, **data}

    return all_data


def load_ticker_data_json(save_json_path: str):

    # get price df
    if save_json_path:
        # read json file if exists
        all_data = processJSON(save_json_path)

    return all_data


if __name__ == '__main__':

    # import ticker data
    import_ticker_data(ticker_file_path=etf_ticker_path,
                       start_date=start_date,
                       end_date=end_date,
                       time_interval='daily',
                       save_json_path=json_path,
                       no_data_json_path=no_data_json_path,
                       save_data_only=True)
