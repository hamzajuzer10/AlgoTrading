from backtest_portfolio_optimisation import volest
from backtest_portfolio_optimisation import data

# data
# symbol = 'JPM'
# bench = '^GSPC'
coin = ['ETH']
bench = ['BTC']
# data_file_path = 'tests\\JPM.csv'
# bench_file_path = 'tests\\BENCH.csv'
estimator = 'HodgesTompkins'
from_date = '2018-04-15'
to_date = '2021-05-23'

# estimator windows
window = 30
windows = [30, 60, 90, 120]
quantiles = [0.25, 0.75]
bins = 100
normed = True

# use the yahoo helper or get coin data to correctly format data
eth_price_data = data.get_coin_data(coin, from_date, to_date)
btc_price_data = data.get_coin_data(bench, from_date, to_date)

# jpm_price_data = data.yahoo_helper(symbol, data_file_path)
# spx_price_data = data.yahoo_helper(bench, bench_file_path)

# initialize class
vol = volest.VolatilityEstimator(
    price_data=eth_price_data,
    estimator=estimator,
    bench_data=btc_price_data
)

# call plt.show() on any of the below...
realized, last = vol.realized_vol()
_, plt = vol.cones(windows=windows, quantiles=quantiles)
_, plt = vol.rolling_quantiles(window=window, quantiles=quantiles)
_, plt = vol.rolling_extremes(window=window)
_, plt = vol.rolling_descriptives(window=window)
_, plt = vol.histogram(window=window, bins=bins, normed=normed)

_, plt = vol.benchmark_compare(window=window)
_, plt = vol.benchmark_correlation(window=window)

# ... or create a pdf term sheet with all metrics in term-sheets/
vol.term_sheet(
    window,
    windows,
    quantiles,
    bins,
    normed
)