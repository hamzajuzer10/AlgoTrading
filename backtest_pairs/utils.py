import os


def write_to_file(path, text_stream, append_mode):
    """Writes to a text file"""

    with open(path, append_mode) as out_file:
        out_file.write(text_stream)


def save_file_path(folder_name:str, filename:str):
    """Outpyts the full filepath and creates the folder if doesnt exist"""

    cwd = os.getcwd()
    model_dir = os.path.join(cwd, folder_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    file_dir = os.path.join(model_dir, filename)

    return file_dir


def create_dict(ticker, max_date=None, min_date=None, n_samples=None,
                johansen_90_p_trace=None, johansen_95_p_trace=None, johansen_99_p_trace=None, johansen_trace_stat=None,
                johansen_90_p_eigen=None, johansen_95_p_eigen=None, johansen_99_p_eigen=None, johansen_eigen_stat=None,
                johansen_eigenvectors=None, johansen_eigenvalue=None, merged_prices_comb_df=None, adf_test_stat=None,
                adf_99_p_stat=None, adf_95_p_stat=None, adf_90_p_stat=None, hurst_exp=None, half_life_=None, sample_pass=False,
                comment=None, save_price_df=True):

    if save_price_df:
        data = {'ticker': ticker,
                'max_date': max_date,
                'min_date': min_date,
                'n_samples': n_samples,
                'johansen_90_p_trace': johansen_90_p_trace,
                'johansen_95_p_trace': johansen_95_p_trace,
                'johansen_99_p_trace': johansen_99_p_trace,
                'johansen_trace_stat': johansen_trace_stat,
                'johansen_90_p_eigen': johansen_90_p_eigen,
                'johansen_95_p_eigen': johansen_95_p_eigen,
                'johansen_99_p_eigen': johansen_99_p_eigen,
                'johansen_eigen_stat': johansen_eigen_stat,
                'johansen_eigenvectors': johansen_eigenvectors,
                'johansen_eigenvalue': johansen_eigenvalue,
                'merged_prices_comb_df': merged_prices_comb_df,
                'adf_test_stat': adf_test_stat,
                'adf_99_p_stat': adf_99_p_stat,
                'adf_95_p_stat': adf_95_p_stat,
                'adf_90_p_stat': adf_90_p_stat,
                'hurst_exp': hurst_exp,
                'half_life_': half_life_,
                'sample_pass': sample_pass,
                'comment': comment}
    else:
        data = {'ticker': ticker,
                'max_date': max_date,
                'min_date': min_date,
                'n_samples': n_samples,
                'johansen_90_p_trace': johansen_90_p_trace,
                'johansen_95_p_trace': johansen_95_p_trace,
                'johansen_99_p_trace': johansen_99_p_trace,
                'johansen_trace_stat': johansen_trace_stat,
                'johansen_90_p_eigen': johansen_90_p_eigen,
                'johansen_95_p_eigen': johansen_95_p_eigen,
                'johansen_99_p_eigen': johansen_99_p_eigen,
                'johansen_eigen_stat': johansen_eigen_stat,
                'johansen_eigenvectors': johansen_eigenvectors,
                'johansen_eigenvalue': johansen_eigenvalue,
                'adf_test_stat': adf_test_stat,
                'adf_99_p_stat': adf_99_p_stat,
                'adf_95_p_stat': adf_95_p_stat,
                'adf_90_p_stat': adf_90_p_stat,
                'hurst_exp': hurst_exp,
                'half_life_': half_life_,
                'sample_pass': sample_pass,
                'comment': comment}

    return data
