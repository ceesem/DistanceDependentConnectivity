__all__ = ["rename_by_layer","build_tables","class_spitter","type_spitter","prep_tables","prep_tables_thresh","final_prep",
"find_orphans","depth_divider"]

import pandas as pd
import numpy as np
from tqdm import tqdm
from .dist import Euc_cell2cell, Rad_cell2cell, Euc_syn2cell, Rad_syn2cell
from .connect_stats import binomial_CI

def rename_by_layer(df,depths,depth_names):
    """
    Rename cell types in any df using their pt_position_y, or depth in the sample, to determine which cortical layer they're in.

    Parameters
    ----------
    df : pandas dataframe
        Must have split positions and 'cell_type'
    depths : A 2D numpy array of shape (len(layers), 2)
        Array of discrete layer bounds.
    depth_names : A 1D numpy array of shape (len(layers),)
        Array consisting of layer names to be used in new cell_type.

    Returns
    -------
    out : pandas dateframe
        with fancy new cell types!

    Examples
    --------
    depths = [[20,40],[40,60]]
    depth_names = ['23','4']
    """
    for i in range(len(depths)):
        df.loc[(df['cell_type']=='BC') & ((df['pt_position_y']*(4/1000))>=depths[i][0])
               & ((df['pt_position_y']*(4/1000))<depths[i][1]), 'cell_type'] = '{0:s}_BC'.format(depth_names[i])
        df.loc[(df['cell_type']=='BPC') & ((df['pt_position_y']*(4/1000))>=depths[i][0])
               & ((df['pt_position_y']*(4/1000))<depths[i][1]), 'cell_type'] = '{0:s}_BPC'.format(depth_names[i])
        df.loc[(df['cell_type']=='MC') & ((df['pt_position_y']*(4/1000))>=depths[i][0])
               & ((df['pt_position_y']*(4/1000))<depths[i][1]), 'cell_type'] = '{0:s}_MC'.format(depth_names[i])
        df.loc[(df['cell_type']=='6P') & ((df['pt_position_y']*(4/1000))>=depths[i][0])
               & ((df['pt_position_y']*(4/1000))<depths[i][1]), 'cell_type'] = '{0:s}_P'.format(depth_names[i])
    df.loc[(df['cell_type']=='23_P'), 'cell_type'] = 'Omitted_P'
    df.loc[(df['cell_type']=='4_P'), 'cell_type'] = 'Omitted_P'
    df.loc[(df['cell_type']=='6_BPC'), 'cell_type'] = '5_BPC'
#     df.loc[(df['cell_type']=='6_BC'), 'cell_type'] = 'BC'
#     df.loc[(df['cell_type']=='6_MC'), 'cell_type'] = 'MC'
    return df

def build_tables(client,pre_df,depth_intervals,depth_names,syndup=None):
    """

    Parameters
    ----------
    client : CAVEclient token
    pre_df : pandas dataframe
        Must have split positions and 'cell_type'
    depths : A 2D numpy array of shape (len(layers), 2)
        Array of discrete layer bounds.
    depth_names : A 1D numpy array of shape (len(layers),)
        Array consisting of layer names to be used in new cell_type.
    syn_dup : Optional
        Determines whether a df will be returned consisting of all synapses, aka duplicate root_ids, and split positions

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
    main,syn,nonsyn = build_tables(client,pre,[[20,40],[40,60]],['23','4'])

    """
    pre_root_id = pre_df.pt_root_id.values[0]
    syn_unfiltered = client.materialize.query_table('synapses_pni_2',
                                                filter_equal_dict={'pre_pt_root_id':pre_root_id},
                                                select_columns=['pre_pt_root_id','post_pt_root_id',
                                                               'size','ctr_pt_position','valid'])
    # if updated, this will change
    correct_soma_table = client.info.get_datastack_info()['soma_table']
    # x, y, and z will have their own columns
    nuclei_unmasked = client.materialize.query_table(correct_soma_table,split_positions=True)
    # new df of just neurons (no glial cells)
    nuclei = nuclei_unmasked.query('cell_type == "neuron"').reset_index(drop=True)
    # new column saying how many neurons have the same root_id
    nuclei['num_soma'] = nuclei.groupby('pt_root_id').transform('count')['valid']
    # mask the df to throw out merged nuclei (same root_id being assigned to multiple neurons)
    mask_nuclei = nuclei['num_soma'] < 2
    nuclei_full = nuclei[mask_nuclei].reset_index(drop=True)
    # grabbing the unique root_id's of single-body neurons
    unique_nuc = np.unique(nuclei_full.pt_root_id)
    # masking the cell type table for only single-body neurons
    soma_full = client.materialize.query_table('allen_soma_coarse_cell_class_model_v1',
                                               filter_in_dict = {'pt_root_id':unique_nuc},
                                               split_positions=True)
    soma_full.loc[soma_full['cell_type'] == '6CT', 'cell_type'] = '6P'
    soma_full.loc[soma_full['cell_type'] == '6IT', 'cell_type'] = '6P'
    soma_full = rename_by_layer(soma_full,depth_intervals,depth_names)
    # masking the synapse table for only single-body neurons. these contain a ton of duplicates
    syn_nuc_dup = syn_unfiltered.query("post_pt_root_id in @unique_nuc").reset_index(drop=True)
    # new column in synapse table = number of synapses per single soma
    syn_nuc_dup['num_syn'] = syn_nuc_dup.groupby('post_pt_root_id').transform('count')['valid']
    syn_nuc_dup = syn_nuc_dup.sort_values(by=['post_pt_root_id']).reset_index(drop=True)
    # renaming bc 'size' is a function and it messes with grouping
    syn_nuc_dup.rename(columns={'size':'sizes'}, inplace=True)
    # dropping duplicates
    syn_nuc = syn_nuc_dup.drop_duplicates(subset='post_pt_root_id', keep='first').reset_index(drop=True)
    # grabbing every synaptic size and position and stacking them into a tuple so that each unique nucleus has a list of syn sizes
    syn_nuc['ctr_pt_position'] = syn_nuc_dup.assign(ctr_pt_position=tuple(syn_nuc_dup.ctr_pt_position)).groupby('post_pt_root_id').ctr_pt_position.apply(list).reset_index(drop=True)
    syn_nuc['sizes'] = syn_nuc_dup.assign(sizes=tuple(syn_nuc_dup.sizes)).groupby('post_pt_root_id').sizes.apply(list).reset_index(drop=True)
    syn_nuc['sum_size'] = syn_nuc.apply(lambda row: sum(row.sizes), axis=1)
    syn_nuc['ave_size'] = syn_nuc.apply(lambda row: row.sum_size / len(row.sizes), axis=1)
    # renaming post_pt_root_id in order to merge correctly
    syn_nuc.rename(columns={'post_pt_root_id':'pt_root_id'}, inplace=True)
    # merge!
    main = pd.merge(soma_full,syn_nuc,on='pt_root_id',how='left')
    # these columns are useless to me
    main = main.drop(columns=['id','valid_x', 'valid_y', 'pt_supervoxel_id'])
    main = main.fillna(0)
    # add new columns for cartesian & radial distance to root_id's
    main['d'] = Euc_cell2cell(pre_df,main)
    main['r'] = Rad_cell2cell(pre_df,main)
    main['d_syn2pre'] = Euc_syn2cell(main,pre_df)
    main['r_syn2pre'] = Rad_syn2cell(main,pre_df)
    main['d_syn2post'] = Euc_syn2cell(main,main)
    main['r_syn2post'] = Rad_syn2cell(main,main)
    # grabbing the unique root_id's of single-body neurons in the synapse table
    unique_syn_nuc = np.unique(syn_nuc.pt_root_id)
    # new tables sorted from main of synaptic targets or non-synaptic neighbors of pre_root_id
    syn = main.query('pt_root_id in @unique_syn_nuc').reset_index(drop=True)
    nonsyn = main.query('pt_root_id not in @unique_syn_nuc').reset_index(drop=True)
    # adding a column for pre that states how many synapses & somas are targetted
    pre_df['num_targets'] = len(syn)
    pre_df['num_syn'] = len(syn_nuc_dup)
    if syndup == None:
        return main,syn,nonsyn
    # unfortunately am lazy and wrote distance function w/o split positions, so this is my workaround
    syn_dup_unfiltered = client.materialize.query_table('synapses_pni_2',
                                            filter_equal_dict={'pre_pt_root_id':pre_root_id},
                                            select_columns=['pre_pt_root_id','post_pt_root_id',
                                                            'size','ctr_pt_position','valid'],
                                            split_positions=True)
    syn_dup_df = syn_dup_unfiltered.query("post_pt_root_id in @unique_nuc").reset_index(drop=True)
    syn_dup_df['num_syn'] = syn_dup_df.groupby('post_pt_root_id').transform('count')['valid']
    syn_dup_df = syn_dup_df.sort_values(by=['post_pt_root_id']).reset_index(drop=True)
    syn_dup_df.rename(columns={'size':'sizes'}, inplace=True)
    syn_dup_df.rename(columns={'post_pt_root_id':'pt_root_id'}, inplace=True)
    maindup = pd.merge(soma_full,syn_dup_df,on='pt_root_id',how='right')
    maindup = maindup.drop(columns=['id','valid_x', 'valid_y', 'pt_supervoxel_id'])
    maindup = maindup.fillna(0)
    syndup = maindup.query('pt_root_id in @unique_syn_nuc').reset_index(drop=True)
    syndup['d'] = Euc_cell2cell(pre_df,syndup)
    syndup['r'] = Rad_cell2cell(pre_df,syndup)
    return main,syn,nonsyn,syndup

def class_spitter(main,df_to_sort):
    classes = np.unique(main.classification_system)
    cellarray = []
    for i in range(len(classes)):
        new = df_to_sort.query(f"classification_system in @classes[{i}]").reset_index(drop=True)
        cellarray.append(new)
    return cellarray

def type_spitter(main,df_to_sort):
    types = np.unique(main.cell_type)
    cellarray = []
    for i in range(len(types)):
        new = df_to_sort.query(f"cell_type in @types[{i}]").reset_index(drop=True)
        cellarray.append(new)
    return cellarray

def prep_tables(main,syn,nonsyn,r_interval,upper_distance_limit,syndup=None):
    main_types = type_spitter(main,main)
    syn_types = type_spitter(main,syn)
    nonsyn_types = type_spitter(main,nonsyn)

    bins = np.array(range(0,upper_distance_limit,r_interval))
    bins = np.append(bins,r_interval+bins[-1])
    f_type,s_type = [],[]
    for j in range(len(main_types)):
        f,s = binomial_CI(main_types[j],bins)
        f_type.append(f)
        s_type.append(s)
    if type(syndup) == pd.DataFrame:
        syndup_types = type_spitter(main,syndup)
        return main_types,syn_types,nonsyn_types,f_type,s_type,syndup_types
    else:
        return main_types,syn_types,nonsyn_types,f_type,s_type

def prep_tables_thresh(main,syn,nonsyn,r_interval,upper_distance_limit,threshold):
    main['thresh'] = main.apply(lambda row: np.min(row.d_syn2post) < threshold, axis=1)
    syn['thresh'] = syn.apply(lambda row: np.min(row.d_syn2post) < threshold, axis=1)
    nonsyn['thresh'] = nonsyn.apply(lambda row: np.min(row.d_syn2post) < threshold, axis=1)
    mask_m = main['thresh'] == True
    mask_s = syn['thresh'] == True
    mask_n = nonsyn['thresh'] == True
    main_thresh = main[mask_m].reset_index(drop=True)
    syn_thresh = syn[mask_s].reset_index(drop=True)
    nonsyn_thresh = nonsyn[mask_n].reset_index(drop=True)

    main_types_thresh = type_spitter(main_thresh,main_thresh)
    syn_types_thresh = type_spitter(main_thresh,syn_thresh)
    nonsyn_types_thresh = type_spitter(main_thresh,nonsyn_thresh)

    bins = np.array(range(0,upper_distance_limit,r_interval))
    bins = np.append(bins,r_interval+bins[-1])
    f_type_thresh,s_type_thresh = [],[]
    for j in range(len(main_types_thresh)):
        f,s = binomial_CI(main_types_thresh[j],bins)
        f_type_thresh.append(f)
        s_type_thresh.append(s)
    return main_thresh,syn_thresh,nonsyn_thresh,main_types_thresh,syn_types_thresh,nonsyn_types_thresh,f_type_thresh,s_type_thresh

def final_prep(main,syn,nonsyn,r_interval,upper_distance_limit,syndup=None,threshold=None):
    main_types,syn_types,syndup_types,nonsyn_types,f_type,s_type = [],[],[],[],[],[]
    main_thresh,syn_thresh,nonsyn_thresh,main_types_thresh,syn_types_thresh,nonsyn_types_thresh,f_type_thresh,s_type_thresh = [],[],[],[],[],[],[],[]
    for i in tqdm(range(len(main))):
        if threshold == None:
            if syndup == None:
                beh = prep_tables(main[i],syn[i],nonsyn[i],r_interval,upper_distance_limit,syndup)
                main_types.append(beh[0])
                syn_types.append(beh[1])
                nonsyn_types.append(beh[2])
                f_type.append(beh[3])
                s_type.append(beh[4])
            else:
                beh = prep_tables(main[i],syn[i],nonsyn[i],r_interval,upper_distance_limit,syndup[i])
                main_types.append(beh[0])
                syn_types.append(beh[1])
                nonsyn_types.append(beh[2])
                f_type.append(beh[3])
                s_type.append(beh[4])
                syndup_types.append(beh[5])
        else:
            bep = prep_tables_thresh(main[i],syn[i],nonsyn[i],r_interval,upper_distance_limit,threshold)
            main_thresh.append(bep[0])
            syn_thresh.append(bep[1])
            nonsyn_thresh.append(bep[2])
            main_types_thresh.append(bep[3])
            syn_types_thresh.append(bep[4])
            nonsyn_types_thresh.append(bep[5])
            f_type_thresh.append(bep[6])
            s_type_thresh.append(bep[7])
    if threshold == None:
        if syndup == None:
            return main_types,syn_types,nonsyn_types,f_type,s_type
        else:
            return main_types,syn_types,nonsyn_types,f_type,s_type,syndup_types
    else:
        return main_thresh,syn_thresh,nonsyn_thresh,main_types_thresh,syn_types_thresh,nonsyn_types_thresh,f_type_thresh,s_type_thresh

def find_orphans(client,pre_df):
    syn_unfiltered = client.materialize.query_table('synapses_pni_2',
                                                filter_equal_dict={'pre_pt_root_id':pre_df.pt_root_id.values[0]})
    correct_soma_table = client.info.get_datastack_info()['soma_table']
    nuclei_unmasked = client.materialize.query_table(correct_soma_table,split_positions=True)
    unique_nuc = np.unique(nuclei_unmasked.pt_root_id)
    orph = syn_unfiltered.query("post_pt_root_id not in @unique_nuc").reset_index(drop=True)
    unique_orphans = np.unique(orph.post_pt_root_id)
    uniq_orph = np.array_split(unique_orphans,len(unique_orphans))
    orph = orph.drop(columns=['id', 'valid', 'pre_pt_supervoxel_id',
                          'post_pt_supervoxel_id', 'pre_pt_position', 'post_pt_position'])
    orph = orph.sort_values(by=['post_pt_root_id']).reset_index(drop=True)
    orph['num_syn'] = orph.groupby('post_pt_root_id')['post_pt_root_id'].transform('count')
    orph.rename(columns={'size':'sizes'}, inplace=True)
    osyn = orph.drop_duplicates(subset='post_pt_root_id', keep='first').reset_index(drop=True)
    osyn['ctr_pt_position'] = orph.assign(ctr_pt_position=tuple(orph.ctr_pt_position)).groupby('post_pt_root_id').ctr_pt_position.apply(list).reset_index(drop=True)
    osyn['sizes'] = orph.assign(sizes=tuple(orph.sizes)).groupby('post_pt_root_id').sizes.apply(list).reset_index(drop=True)
    osyn['sum_size'] = osyn.apply(lambda row: sum(row.sizes), axis=1)
    osyn['ave_size'] = osyn.apply(lambda row: row.sum_size / len(row.sizes), axis=1)
    osyn['d'] = Euc_syn2cell(osyn,pre_df)
    osyn['r'] = Rad_syn2cell(osyn,pre_df)
    osyn['d_ave'] = osyn.apply(lambda row: sum(row.d)/len(row.d), axis=1)
    osyn['d_range'] = osyn.apply(lambda row: np.max(row.d) - np.min(row.d), axis=1)
    osyn['r_ave'] = osyn.apply(lambda row: sum(row.r)/len(row.r), axis=1)
    osyn['r_range'] = osyn.apply(lambda row: np.max(row.r) - np.min(row.r), axis=1)
    return osyn,uniq_orph,len(orph)

def depth_divider(depths,main,syn,nonsyn,r_interval,upper_distance_limit):
    d_main,d_syn,d_nonsyn = [],[],[]
    for i in range(len(depths)):
        dep_m,dep_syn,dep_non = [],[],[]
        for j in range(len(syn)):
            m0 = main[j][((main[j]['pt_position_y']*(4/1000))<depths[i][1]) & ((main[j]['pt_position_y']*(4/1000))>depths[i][0])].reset_index(drop=True)
            m1 = syn[j][((syn[j]['pt_position_y']*(4/1000))<depths[i][1]) & ((syn[j]['pt_position_y']*(4/1000))>depths[i][0])].reset_index(drop=True)
            m2 = nonsyn[j][((nonsyn[j]['pt_position_y']*(4/1000))<depths[i][1]) & ((nonsyn[j]['pt_position_y']*(4/1000))>depths[i][0])].reset_index(drop=True)
            dep_m.append(m0)
            dep_syn.append(m1)
            dep_non.append(m2)
        d_main.append(dep_m)
        d_syn.append(dep_syn)
        d_nonsyn.append(dep_non)
    d_main_types,d_syn_types,d_nonsyn_types = [],[],[]
    for i in range(len(depths)):
        ddm,dds,ddn = [],[],[]
        for j in range(len(d_syn[i])):
            tm = type_spitter(main[j],d_main[i][j])
            ts = type_spitter(main[j],d_syn[i][j])
            tn = type_spitter(main[j],d_nonsyn[i][j])
            ddm.append(tm)
            dds.append(ts)
            ddn.append(tn)
        d_main_types.append(ddm)
        d_syn_types.append(dds)
        d_nonsyn_types.append(ddn)
    bins = np.array(range(0,upper_distance_limit,r_interval))
    bins = np.append(bins,r_interval+bins[-1])
    d_f_type,d_s_type = [],[]
    for i in range(len(depths)):
        ddff,ddss = [],[]
        for j in range(len(d_main_types[i])):
            cellf,cells = [],[]
            for k in range(len(d_main_types[i][j])):
                f,s = binomial_CI(d_main_types[i][j][k],bins)
                cellf.append(f)
                cells.append(s)
            ddff.append(cellf)
            ddss.append(cells)
        d_f_type.append(ddff)
        d_s_type.append(ddss)
    return d_main,d_syn,d_nonsyn,d_main_types,d_syn_types,d_nonsyn_types,d_f_type,d_s_type