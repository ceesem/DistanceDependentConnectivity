__all__ = ["build_tables","class_spitter","type_spitter","prep_tables","prep_tables_thresh","find_orphans"]

import pandas as pd
import numpy as np
from .dist import Euc_cell2cell, Rad_cell2cell, Euc_syn2cell, Rad_syn2cell
from .connect_stats import binomial_CI

def build_tables(client,pre_df):
    pre_root_id = pre_df.pt_root_id.values[0]
    syn_unfiltered = client.materialize.query_table('synapses_pni_2',
                                                filter_equal_dict={'pre_pt_root_id':pre_root_id})
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
    main = main.drop(columns=['id_x', 'id_y', 'valid_x', 'valid_y', 'pt_supervoxel_id', 'pre_pt_supervoxel_id',
                          'post_pt_supervoxel_id', 'pre_pt_position', 'post_pt_position'])
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
    return main,syn,nonsyn

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

def prep_tables(main,syn,nonsyn,r_interval,upper_distance_limit):
    main_types = type_spitter(main,main)
    syn_types = type_spitter(main,syn)
    nonsyn_types = type_spitter(main,nonsyn)

    bins = np.array(range(0,upper_distance_limit,r_interval))
    f_type,s_type = [],[]
    for j in range(len(main_types)):
        f,s = binomial_CI(main_types[j],bins)
        f_type.append(f)
        s_type.append(s)
    return main_types,syn_types,nonsyn_types,f_type,s_type

def prep_tables_thresh(main,syn,r_interval,upper_distance_limit,threshold):
    main['thresh'] = main.apply(lambda row: np.min(row.d_syn2post) < threshold, axis=1)
    syn['thresh'] = syn.apply(lambda row: np.min(row.d_syn2post) < threshold, axis=1)
    mask_m = main['thresh'] == True
    mask_s = syn['thresh'] == True
    main_thresh = main[mask_m].reset_index(drop=True)
    syn_thresh = syn[mask_s].reset_index(drop=True)

    main_types_thresh = type_spitter(main_thresh,main_thresh)
    syn_types_thresh = type_spitter(main_thresh,syn_thresh)

    bins = np.array(range(0,upper_distance_limit,r_interval))
    f_type_thresh,s_type_thresh = [],[]
    for j in range(len(main_types_thresh)):
        f,s = binomial_CI(main_types_thresh[j],bins)
        f_type_thresh.append(f)
        s_type_thresh.append(s)
    return main_thresh,syn_thresh,main_types_thresh,syn_types_thresh,f_type_thresh,s_type_thresh

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