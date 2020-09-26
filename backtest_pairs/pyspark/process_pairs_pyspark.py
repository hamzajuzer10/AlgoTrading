from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, StructField, StructType, StringType, IntegerType, DateType, FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType
import pandas as pd
from time import time
import numpy as np
from backtest_pairs.process_pairs import calculate_coint_results

master = 'local[4]'
csv_path = "C:\\Users\\hamzajuzer\\Documents\\Algorithmic Trading\AlgoTradingv1\\backtest_pairs\\pyspark\\etf_tickers_test.csv"
save_path = "C:\\Users\\hamzajuzer\\Documents\\Algorithmic Trading\AlgoTradingv1\\backtest_pairs\\pyspark\\etf_tickers_test_results.csv"
min_period_yrs = 1.5
max_half_life = 30 # in time interval units
min_half_life = 2 # in time interval units

if __name__ == '__main__':

    # Create Spark session
    conf = SparkConf().setMaster(master)
    spark = SparkSession.builder.config(conf=conf) \
        .getOrCreate()

    # Enable Arrow optimization and fallback if there is no Arrow installed
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")

    # Read the dataframe from a csv
    schema = StructType([
        StructField('formatted_date', DateType(), True),
        StructField('x_0', FloatType(), True),
        StructField('x_1', FloatType(), True),
        StructField('combination', StringType(), True)
    ])

    df = spark.read.csv(csv_path, header=True, schema=schema, dateFormat="dd/MM/yyyy")

    schema_output = StructType([
        StructField('ticker', ArrayType(StringType(), True), True),
        StructField('max_date', DateType(), True),
        StructField('min_date', DateType(), True),
        StructField('n_samples', IntegerType(), True),
        StructField('johansen_90_p_trace', FloatType(), True),
        StructField('johansen_95_p_trace', FloatType(), True),
        StructField('johansen_99_p_trace', FloatType(), True),
        StructField('johansen_trace_stat', FloatType(), True),
        StructField('johansen_90_p_eigen', FloatType(), True),
        StructField('johansen_95_p_eigen', FloatType(), True),
        StructField('johansen_99_p_eigen', FloatType(), True),
        StructField('johansen_eigen_stat', FloatType(), True),
        StructField('johansen_eigenvectors', ArrayType(FloatType(), True), True),
        StructField('johansen_eigenvalue', FloatType(), True),
        StructField('adf_test_stat', FloatType(), True),
        StructField('adf_99_p_stat', FloatType(), True),
        StructField('adf_95_p_stat', FloatType(), True),
        StructField('adf_90_p_stat', FloatType(), True),
        StructField('hurst_exp', FloatType(), True),
        StructField('half_life_', FloatType(), True),
        StructField('sample_pass', IntegerType(), True),
        StructField('comment', StringType(), True)
    ])

    @pandas_udf(schema_output, PandasUDFType.GROUPED_MAP)
    def process_pairs_spark(df):

        ticker = df['combination'].iloc[0].split("_")
        ticker_col = [col for col in df if col.startswith('x')]

        merged_prices_comb_df = df.loc[:, df.columns != 'combination']
        merged_prices_comb_df.set_index('formatted_date', inplace=True)

        result_dict = calculate_coint_results(merged_prices_comb_df=merged_prices_comb_df,
                                              ticker=ticker,
                                              min_period_yrs=min_period_yrs,
                                              max_half_life=max_half_life,
                                              min_half_life=min_half_life,
                                              save_price_df=False,
                                              save_all=True,
                                              print_verbose=False,
                                              print_file=False,
                                              alt_cols=ticker_col)

        # return results dataframe
        results_df = pd.DataFrame()
        results_df = results_df.append(result_dict, ignore_index=True)
        return results_df


    t_0 = time()

    df_map = df.groupby("combination").apply(process_pairs_spark)
    df_map.show()
    results_df_p = df_map.toPandas()
    results_df_p.to_csv(save_path, mode='w', index=True)

    print(np.round(time() - t_0, 3), "seconds elapsed...")