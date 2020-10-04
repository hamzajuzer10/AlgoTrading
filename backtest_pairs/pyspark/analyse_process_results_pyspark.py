import pandas as pd
import numpy as np
import ast
from backtest_pairs.process_pairs import build_price_df
from backtest_pairs.yfinance_connector import import_ticker_data, load_ticker_data_json
from backtest_pairs.utils import save_file_path

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)

results_save_path = "C:\\Users\\hamzajuzer\\Documents\\Algorithmic Trading\AlgoTradingv1\\backtest_pairs\\pyspark\\etf_tickers_results.csv"
json_path = "C:\\Users\\hamzajuzer\\Documents\\Algorithmic Trading\AlgoTradingv1\\backtest_pairs\\data\\etf_tickers.json"

max_date = '2020-05-29'


def build_merged_prices_comb_df(results_df: pd.DataFrame, ticker_data_path: str):

    # Load ticker data from path
    ticker_data = load_ticker_data_json(ticker_data_path)

    # Build price df
    ticker_data = build_price_df(ticker_data)

    # For each row in results_df
    results_df.reset_index(drop=True, inplace=True)
    merged_prices_index_list = []
    merged_prices_df_list = []
    for index, row in results_df.iterrows():

        print('Processing combination {a}/{b}: {c}'.format(a=index, b=len(results_df), c=row['ticker']))
        merged_prices_comb_df = pd.DataFrame()
        for j in row['ticker']:

            merged_prices_comb_df = pd.merge(merged_prices_comb_df, ticker_data[j]['price_df'], left_index=True,
                                             right_index=True, how='outer')

        merged_prices_comb_df.dropna(inplace=True)

        # Add merged_prices
        merged_prices_index_list.append(index)
        merged_prices_df_list.append(merged_prices_comb_df)

    # Create merged prices dataframe
    merged_prices_df = pd.DataFrame({"index": merged_prices_index_list, "merged_prices_comb_df": merged_prices_df_list})
    merged_prices_df.set_index("index", drop=True, inplace=False)
    results_df = pd.merge(results_df, merged_prices_df, left_index=True, right_index=True, how="inner")

    return results_df


if __name__== '__main__':

    # read ticker results
    print('Reading processed ticker results')
    coint_ticker_results = pd.read_csv(results_save_path)
    coint_ticker_results = coint_ticker_results.drop(coint_ticker_results.columns[0], axis=1)

    # Filter on only positive results
    print('Filtering on positive ticker results')
    coint_ticker_pos_results = coint_ticker_results[coint_ticker_results['sample_pass'] == 1]

    # Filter on results where max date is not current max date - maybe its not active anymore
    print('Filtering on results where max date is not current max date')
    coint_ticker_pos_results = coint_ticker_pos_results[coint_ticker_pos_results['max_date'] >= max_date]

    # Build merged_prices_comb_df for each ticker combination
    print('Building merged priced dataframe')
    coint_ticker_pos_results['ticker'] = coint_ticker_pos_results.apply(lambda x: ast.literal_eval(x['ticker']), axis=1)
    coint_ticker_pos_results = build_merged_prices_comb_df(coint_ticker_pos_results, json_path)

    # saving valid ticker combinations
    print('Saving results')
    coint_ticker_pos_results.to_pickle(save_file_path(folder_name='coint_results', filename='valid_coint_results_df.pkl',
                                                      wd='C:\\Users\\hamzajuzer\\Documents\\Algorithmic Trading\\AlgoTradingv1\\backtest_pairs'))