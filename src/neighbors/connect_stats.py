__all__ = ["binomial_CI","pmax_type"]

import numpy as np

def binomial_CI(df, bins):
    fail = []
    success = []
    for i in range(len(bins)-1):
        # masking for specific distance bin
        masked_df = df[(df['r'] > bins[i]) & (df['r'] < bins[i+1])].reset_index(drop=True)
        # starting the counter
        f,s = 0,0
        for j in range(len(masked_df)):
            if masked_df['pre_pt_root_id'][j] == 0:
                f += 1
            else:
                s += 1
        # if there are zero cells in masked_df, 0's are appended
        fail.append(f)
        success.append(s)
    return np.array(fail),np.array(success)

def pmax_type(s_type,f_type):
    pmax,bin_ind,typemax = [],[],[]
    for j in range(len(s_type)):
        p = np.around(np.nan_to_num((s_type[j]/(f_type[j]+s_type[j]))),4)
        b = np.where(p == np.max(p))[0]
        pmax.append(np.max(p))
        bin_ind.append(b)
    # for j in range(len(s_type)):
    #     if pmax[j] == np.max(pmax):
    #         typemax.append([pmax[j],j])
    #     else:
    #         continue
    return pmax#,bin_ind,typemax

def probfunct(param,df):
    d = df['r']
    p = param[0] * np.exp(-((d-param[2])**2)/(2*(param[1]**2)))
    return p

def log_likelihood(param,syn,nonsyn):
    conn = np.sum(np.log(probfunct(param,syn)))
    unc = np.sum(np.log(1. - probfunct(param,nonsyn)))
    l = conn+unc
    return -l