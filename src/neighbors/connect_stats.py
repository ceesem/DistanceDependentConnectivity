__all__ = ["binomial_CI","pmax_type"]

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize

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

def GaussianMLE(pre,s_type,f_type,syn_types,nonsyn_types,main,threshold=None):
    """

    Parameters
    ----------
    pre_df : pandas dataframe
        Must have split positions and 'cell_type'

    Returns
    -------
    out : 3 pandas dataframes if syndup=False, 4 if True
        first is "main", where all confirmed somas are listed, connected or not.
        second is "syn", a subset of main, only connected somas
        third is "nonsyn", a subset of main, only if unconnected to pre-syn
        fourth (optional) is "syndup", where every synapse has its own row, resulting in duplicate rows of the same soma

    Examples
    --------
    main,syn,nonsyn,syndup = build_tables(client,pre,[[20,40],[40,60]],['23','4'],syndup)

    """
    # change this if you dont like these guesses!
    mu = 0
    sigs = 100
    pguess = .5

    pmax_simple = []
    for i in range(len(pre)):
        presynp = []
        for j in range(len(pre[i])):
            presynp.append(pmax_type(s_type[i][j],f_type[i][j]))
        pmax_simple.append(presynp)

    allsyns,allnons = [],[]
    for i in range(len(pre)):
        syntype,nontype = [],[]
        for k in range(len(syn_types[i][0])):
            syncell,noncell = [],[]
            for j in range(len(syn_types[i])):
                syncell.append(syn_types[i][j][k])
                noncell.append(nonsyn_types[i][j][k])
            syntype.append(pd.concat(syncell,ignore_index=True))
            nontype.append(pd.concat(noncell,ignore_index=True))
        allsyns.append(syntype)
        allnons.append(nontype)

    r_comb = []
    for i in range(len(pre)):
        rpretype = []
        for j in range(len(allsyns[i])):
            sigs = 100
            pguess = .5
            init_guess = [pguess,sigs,mu]
            r = minimize(fun=log_likelihood,x0=init_guess,bounds=[(0,1.),(0,200),(0,100)],
                        method='nelder-mead',options={'maxfev':1000},args=(allsyns[i][j],allnons[i][j]))
            rpretype.append(r)
        r_comb.append(rpretype)
    bad = []
    for i in range(len(r_comb)):
        for j in range(len(r_comb[i])):
            if r_comb[i][j].success == False:
                bad.append([i,j])
            else:
                continue
    if len(bad) > 0:
        print("The following pre-syn types failed:")
        for i in range(len(bad)):
            print(r_comb[bad[i][0]][bad[i][1]][bad[i][2]])

    res_comb = []
    pmax_comb,sigs_comb,moo_comb = [],[],[]
    for i in range(len(r_comb)):
        pp,mm,ss = [],[],[]
        for j in range(len(r_comb[i])):
            pp.append(r_comb[i][j].x[0])
            ss.append(r_comb[i][j].x[1])
            mm.append(r_comb[i][j].x[2])
        pmax_comb.append(pp)
        sigs_comb.append(ss)
        moo_comb.append(mm)
    res_comb.append(pmax_comb)
    res_comb.append(sigs_comb)
    res_comb.append(moo_comb)

    nconn_comb,nprobe_comb = [],[]
    for i in range(len(syn_types)):
        scell,fcell = [],[]
        for k in range(len(syn_types[i][0])):
            stype,ftype = [],[]
            for j in range(len(syn_types[i])):
                stype.append(np.sum(s_type[i][j][k]))
                ftype.append(np.sum(f_type[i][j][k])+np.sum(s_type[i][j][k]))
            scell.append(np.sum(stype))
            fcell.append(np.sum(ftype))
        nconn_comb.append(scell)
        nprobe_comb.append(fcell)

    target_list = np.unique(main[0][0].cell_type)

    results = []
    for i in range(len(pre)):
        rpresyn = []
        for j in tqdm(range(len(pre[i]))):
            rcell = []
            pre_pmax, pre_sig = [],[]
            for k in range(len(syn_types[i][j])):
                init_guess = [pmax_simple[i][j][k],sigs,mu]
                r = minimize(fun=log_likelihood,x0=init_guess,bounds=[(0,1.),(0,200),(0,100)],
                            method='nelder-mead',options={'maxfev':1000},args=(syn_types[i][j][k],nonsyn_types[i][j][k]))
                if r.x[0] == 0.:
                    pmaxx = 0.0001
                else:
                    pmaxx = np.around(r.x[0],4)
                pre_pmax.append(pmaxx)
                if threshold == None:
                    pre[i][j][target_list[k]] = pmaxx
                else:
                    pre[i][j][target_list[k]+'_thresh'] = pmaxx
                if r.x[0] > 0.001:
                    pre_sig.append(np.around(float(r.x[1]+r.x[2]),4))
                else:
                    pre_sig.append(0.0)
                rcell.append(r)
            if threshold == None:
                pre[i][j]['pmax'] = [pre_pmax]
                pre[i][j]['sigma_extent'] = [pre_sig]
            else:
                pre[i][j]['pmax_thresh'] = [pre_pmax]
                pre[i][j]['sigma_extent_thresh'] = [pre_sig]
            rpresyn.append(rcell)
        results.append(rpresyn)
    bad = []
    for i in range(len(results)):
        for j in range(len(results[i])):
            for k in range(len(results[i][j])):
                if results[i][j][k].success == False:
                    bad.append([i,j,k])
                else:
                    continue
    if len(bad) > 0:
        print("The following cells failed:")
        for i in range(len(bad)):
            print(results[bad[i][0]][bad[i][1]][bad[i][2]])

    res = []
    pmax,sigs,moo = [],[],[]
    for i in range(len(results)):
        prep,pres,prem = [],[],[]
        for k in range(len(results[i][0])):
            pp,mm,ss = [],[],[]
            for j in range(len(results[i])):
                pp.append(results[i][j][k].x[0])
                ss.append(results[i][j][k].x[1])
                mm.append(results[i][j][k].x[2])
            prep.append(pp)
            pres.append(ss)
            prem.append(mm)
        pmax.append(prep)
        sigs.append(pres)
        moo.append(prem)
    res.append(pmax)
    res.append(sigs)
    res.append(moo)

    nconn,nprobe = [],[]
    for i in range(len(syn_types)):
        prenc,prenp = [],[]
        beb = pd.concat(pre[i],ignore_index=True).sort_values(by='pt_position_y')
        for j in beb.index.values:
            nc,npr = [],[]
            for k in range(len(syn_types[i][j])):
                nc.append(np.sum(s_type[i][j][k]))
                npr.append(np.sum(f_type[i][j][k])+np.sum(s_type[i][j][k]))
            prenc.append(nc)
            prenp.append(npr)
        nconn.append(prenc)
        nprobe.append(prenp)

    precat = []
    for i in range(len(pre)):
        pppp = pd.concat(pre[i],ignore_index=True).sort_values(by='pt_position_y').reset_index(drop=True)
        precat.append(pppp)
    cat = pd.concat(precat,ignore_index=True).reset_index(drop=True)

    return res_comb,nconn_comb,nprobe_comb,res,nconn,nprobe,precat,cat