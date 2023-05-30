import pandas as pd
import numpy as np
from tqdm import tqdm
from .dist import radial_distance, euclidean_distance
from .connect_stats import binomial_ci


def _compute_soma_distances(
    soma_position,
    target_df,
    soma_position_column="pt_position",
    radial_function=None,
    euclidean_function=None,
):
    targ_xyz = np.vstack(target_df[soma_position_column].values)
    if radial_function is None:
        target_df["r"] = radial_distance(soma_position, targ_xyz)
    else:
        target_df["r"] = radial_function(soma_position, targ_xyz)

    if euclidean_function is None:
        target_df["d"] = euclidean_distance(soma_position, targ_xyz)
    else:
        target_df["d"] = euclidean_function(soma_position, targ_xyz)
    return target_df


def _compute_synapse_edgelist(
    syn_df,
    cell_type_merge_column_syn_table,
    num_syn_column="num_syn",
):
    syn_df[num_syn_column] = syn_df.groupby(cell_type_merge_column_syn_table).transform(
        "count"
    )[syn_df.columns[0]]

    syn_df = syn_df.drop_duplicates(
        subset=cell_type_merge_column_syn_table, keep="first"
    ).reset_index(drop=True)
    return syn_df


def build_tables(
    soma_position,
    syn_df,
    target_df,
    cell_type_column="cell_type",
    cell_type_merge_column_syn_table="post_pt_root_id",
    cell_type_merge_column_ct_table="pt_root_id",
    soma_position_column="pt_position",
    filter_cell_types=None,
    radial_function=None,
    euclidean_function=None,
    num_syn_column="num_syn",
):
    if cell_type_column in syn_df.columns:
        raise ValueError(
            f'Trying to merge cell types into table with cell type column "{cell_type_column}" already present'
        )
    if filter_cell_types is not None:
        target_df.query(f"{cell_type_column} in @filter_cell_types", inplace=True)

    target_df = _compute_soma_distances(
        soma_position,
        target_df,
        soma_position_column=soma_position_column,
        radial_function=radial_function,
        euclidean_function=euclidean_function,
    )

    syn_df = _compute_synapse_edgelist(
        syn_df, cell_type_merge_column_syn_table, num_syn_column=num_syn_column
    )
    if isinstance(soma_position_column, str):
        soma_position_column = [soma_position_column]
    target_df = (
        target_df[
            [cell_type_merge_column_ct_table, 'r', 'd', cell_type_column] + soma_position_column
        ]
        .merge(
            syn_df[[cell_type_merge_column_syn_table, num_syn_column]],
            left_on=cell_type_merge_column_ct_table,
            right_on=cell_type_merge_column_syn_table,
            how="left",
        )
        .drop(columns=cell_type_merge_column_syn_table)
        .fillna(0)
        .reset_index(drop=True)
    )

    return target_df



def compute_confidence_intervals(
    target_df,
    bin_width=15,
    upper_distance_limit=400,
    cell_type_column="cell_type",
    distance_column='r',
):
    bins = np.array(range(0, upper_distance_limit+bin_width, bin_width))
    fs, ss = {}, {}
    for jj in target_df[cell_type_column].unique():
        df = target_df.query(f"{cell_type_column} == @jj")
        f, s = binomial_ci(df, bins=bins, distance_column=distance_column)
        fs[jj] = f
        ss[jj] = s
    return fs, ss, bins
