import pandas as pd
import numpy as np


def kalman_filter(y_df: pd.DataFrame, x_df: pd.DataFrame, init_beta_weights, epsilon_df=None):

    # y is the price of the normalised security in the portfolio
    # x will be the prices of other securities in the portfolio over time (t,dim(x)) array (will need to add a constant)
    # init_beta_weights will be the initial beta weights of the other securities which we get from Johansen
    # epsilon is a 1d array of y-x*init_beta_weights

    # convert dataframe to array
    y = y_df.values
    x = x_df.values
    epsilon = epsilon_df.values if epsilon_df is not None else None

    # initialize
    x = np.append(x, np.ones([x.shape[0],1]), axis=1) # append constant to x
    dim = x.shape[1] # get the dim of x array
    Q = np.zeros(y.shape) # measurement prediction error variance
    sqrt_Q= np.zeros(y.shape)
    e = np.zeros(y.shape) # measurement prediction error
    y_hat = np.zeros(y.shape) # measurement prediction
    P = np.zeros(shape=(dim, dim))
    beta_weights = np.zeros(shape=(y.shape[0], dim))
    delta = 0.0001
    Vw = delta/(1-delta)*np.eye(dim)
    Ve = 0.001
    R = None

    # initalize beta to init_beta_weights and 0 constant
    beta_weights[0,:] = np.append(init_beta_weights, np.mean(epsilon) if epsilon is not None else 0)

    # loop
    for t in range(0, y.shape[0]):

        if t > 0:
            beta_weights[t, :] = beta_weights[t-1,:]
            R = P + Vw
        else:
            R = np.zeros((dim, dim))

        # Compute y_hat
        y_hat[t] = x[t,:].dot(beta_weights[t,:])

        Q[t] = x[t,:].dot(R).dot(x[t,:].T) + Ve

        sqrt_Q[t] = np.sqrt(Q[t])

        e[t] = y[t] - y_hat[t]
        K = R.dot(x[t,:].T) / Q[t]
        beta_weights[t,:] = beta_weights[t,:] + K.flatten() * e[t]
        P = R - K * x[t,:].dot(R)


    # return dataframe of beta weights, errors and sqrt_q
    beta_cols = x_df.columns.to_list()
    beta_cols = beta_cols + ['const']
    pre_str = '_beta'
    beta_cols = [s + pre_str for s in beta_cols]
    beta_weights_df = pd.DataFrame(data=beta_weights, index=x_df.index.to_list(), columns=beta_cols)
    e_df = pd.DataFrame(data=e, index=x_df.index.to_list(), columns=['e'])
    sqrt_Q_df = pd.DataFrame(data=sqrt_Q, index=x_df.index.to_list(), columns=['sqrt_q'])

    return beta_weights_df, e_df, sqrt_Q_df