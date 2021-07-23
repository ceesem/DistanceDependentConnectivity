__all__ = ["binomial_CI"]

import numpy as np

def binomial_CI(df, bins):
    fail = []
    success = []
    # no cells are less than 0 microns away, and this makes sure my arrays are the same size
    fail.append(0)
    success.append(0)
    for i in range(len(bins)-1):
        # masking for specific distance bin
        masked_df = df[(df['r'] < bins[i+1]) & (df['r'] > bins[i])].reset_index(drop=True)
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