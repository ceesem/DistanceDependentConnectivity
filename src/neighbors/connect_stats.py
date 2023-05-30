import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm

def binomial_ci(df, bins, distance_column='r'):
    fail = []
    success = []
    df = df.reset_index(drop=True)
    df['bin_num'] = pd.cut(
        df[distance_column],
        bins,
        labels=False,
        right=True,
    )
    for ii in np.sort(df['bin_num'].dropna().unique()):
        masked_df = df.query('bin_num == @ii').reset_index(drop=True)
        fail.append(np.sum(masked_df['num_syn'] == 0))
        success.append(np.sum(masked_df['num_syn'] > 0))
    return np.array(fail),np.array(success)

def pmax(success, fail):
    return np.max(np.nan_to_num( success / (success+fail) ))

def fit_loglikelihood(target_conn, target_nonconn, pm_init=0.5, mu_init=0, sig_init=100):
    init_guess = [pm_init, sig_init, mu_init]
    r = minimize(
        fun=neg_log_likelihood,
        x0=init_guess,
        bounds=[(0, 1.0), (40, 200), (0, 100)],
        method="nelder-mead",
        options={"maxfev": 1000},
        args=(
            target_conn,
            target_nonconn,
        ),
    )
    return r.x

def compute_binned_errorbars(s, f, method='wilson'):
    errorbars = sm.stats.proportion_confint(
        s, nobs=(s+f), method=method
    )
    probability = np.nan_to_num(s / (s+f))
    return probability, (np.abs(probability - errorbars[0]), np.abs(errorbars[1] - probability))


def probfunct(param,d):
    p = param[0] * np.exp(-((d-param[2])**2)/(2*(param[1]**2)))
    return p

def log_likelihood(param,syn,nonsyn, distance_column='r'):
    conn = np.sum(np.log(probfunct(param,syn[distance_column])))
    unc = np.sum(np.log(1. - probfunct(param,nonsyn[distance_column])))
    l = conn+unc
    return l

def neg_log_likelihood(param,syn,nonsyn, distance_column='r'):
    conn = np.sum(np.log(probfunct(param,syn[distance_column])))
    unc = np.sum(np.log(1. - probfunct(param,nonsyn[distance_column])))
    l = conn+unc
    return -l

