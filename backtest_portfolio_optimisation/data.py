import pandas
from backtest_portfolio_optimisation.optimise_portfolio import fetch_all

def get_coin_data(coin, start_date, end_date):

    coin_history = fetch_all(coins=coin, start_date=start_date, end_date=end_date)
    data = coin_history[coin[0]]
    data = data[['high', 'low', 'open', 'close']].rename(columns={
             'high': 'High', 'low': 'Low', 'open': 'Open', 'close': 'Close'
         })
    data.symbol = coin[0]


    # try:
    #     data = pandas.read_csv(
    #         data_path,
    #         parse_dates=['Date'],
    #         index_col='Date',
    #         usecols=[
    #             'Date',
    #             'Open',
    #             'High',
    #             'Low',
    #             'Adj Close',
    #         ],
    #         *args
    #     ).rename(columns={
    #         'Adj Close': 'Close'
    #     })
    #
    # except Exception as e:
    #     raise e
    #
    #
    return data



def yahoo_helper(symbol, data_path, *args):
    """
    Returns DataFrame/Panel of historical stock prices from symbols, over date
    range, start to end. 
    
    Parameters
        ----------
        symbol : string
            Single stock symbol (ticker)
        data_path: string
            Path to Yahoo! historical data CSV file
        *args:
            Additional arguments to pass to pandas.read_csv
    """

    try:
        data = pandas.read_csv(
            data_path,
            parse_dates=['Date'],
            index_col='Date',
            usecols=[
                'Date',
                'Open',
                'High',
                'Low',
                'Adj Close',
            ],
            *args
        ).rename(columns={
            'Adj Close': 'Close'
        })

    except Exception as e:
        raise e

    data.symbol = symbol
    return data
