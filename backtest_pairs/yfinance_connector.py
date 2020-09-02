import pandas as pd
from yahoofinancials import YahooFinancials
import warnings
import sys


def import_ticker_data(start_date: str, end_date: str, time_interval: str, ticker_file_path: str = None, tickers: list = None):

    # read csv file with ticker data
    if ticker_file_path:
        ticker_df = pd.read_csv(ticker_file_path)
        ticker_list = ticker_df['Symbol'].tolist()
    elif tickers:
        ticker_list = tickers
    else:
        warnings.warn('No file path or list of tickers provided')
        sys.exit(0)

    yahoo_financials = YahooFinancials(ticker_list)

    # download data
    data = yahoo_financials.get_historical_price_data(start_date=start_date,
                                                      end_date=end_date,
                                                      time_interval=time_interval)

    # get the prices as a dataframe
    for ticker in list(data):

        # if prices is a key
        try:
            price_df = pd.DataFrame(data[ticker]['prices'])
            price_df = price_df.drop('date', axis=1).set_index('formatted_date')
            price_df.index = pd.to_datetime(price_df.index, format='%Y-%m-%d %H:%M:%S')
            price_df = price_df[['adjclose']]
            price_df.rename(columns={'adjclose': ticker}, inplace=True)
            data[ticker]['price_df'] = price_df
        except KeyError:
            warnings.warn('Price data not available for {}!'.format(ticker))
            warnings.warn('Removing {} from data list'.format(ticker))
            del data[ticker]

    if len(data.keys()) == 0:
        warnings.warn('No ticker price data available for requested tickers!')
        sys.exit(0)

    # TODO: research if Xccy needs to be the same for all ticker prices - currently keeping Xccy in base units

    return data