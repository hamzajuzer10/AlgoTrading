import json
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, date


# fetch coin history, supported timescales include daily and hourly
def fetch_all(coins, days_ago_to_fetch=None, start_date=None, end_date=None, timescale='daily'):

    coin_history = {}
    for coin in coins:
        coin_history[coin] = fetch_history(coin, days_ago_to_fetch, start_date, end_date, timescale)

    return coin_history


def fetch_history(coin, days_ago_to_fetch=None, start_date=None, end_date=None, timescale='daily'):

    if start_date and end_date:
        start_date_d = datetime.strptime(start_date, '%Y-%m-%d').date()
        days_ago_to_fetch = (date.today() - start_date_d).days

    elif not days_ago_to_fetch:

        raise ValueError("No argument for days ago to fetch, start date or end date included in fetch history")

    # convert fetch period to hours if timescale is hourly
    if timescale == 'hourly':
        days_ago_to_fetch = days_ago_to_fetch*24

    # api has a limit of 2000 so raise a Value error if the limit is breached
    if days_ago_to_fetch > 2000:

        raise ValueError("Data request is greater than the maximum data points of 2000 periods")

    period = 'day' if timescale is 'daily' else 'hour'

    endpoint_url = "https://min-api.cryptocompare.com/data/histo{}?fsym={}&tsym=USD&limit={:d}".format(period, coin,
                                                                                                       days_ago_to_fetch)
    res = requests.get(endpoint_url)
    hist = pd.DataFrame(json.loads(res.content)['Data'])
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')

    if start_date and end_date:
        hist = filter_history_by_date(hist, start_date, end_date)

    return hist


def filter_history_by_date(hist, start_date, end_date):
    start_date_d = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date_d = datetime.strptime(end_date, '%Y-%m-%d').date()

    result = hist[(hist.index.date >= start_date_d) & (hist.index.date <= end_date_d)]
    return result
