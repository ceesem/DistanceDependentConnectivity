import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns
import statsmodels as sm
import warnings
from tqdm import tqdm
from scipy.optimize import minimize
from .connect_stats import probfunct, log_likelihood

def threepanels_pertype(
    main,
    pre,
    syn_types,
    nonsyn_types,
    s_type,
    f_type,
    r_interval,
    upper_distance_limit,
    filename,
    MLEresults=True,
    display=False,
):
    warnings.filterwarnings("ignore")
    unique_types = np.unique(main.cell_type)
    fig, ax = plt.subplots(len(unique_types), 3)
    fig.set_size_inches(len(unique_types) * 0.67, len(unique_types) * 3.5)
    pmax = pmax_type(s_type, f_type)
    if MLEresults == True:
        mu = 0
        results = []
        sigs = 100
        # if pre.cell_type.values[0] == any(['BC','BPC','MC']):
        #     sigs = [95,95,95,95,95,95,125,125,125,95]
        # else:
        #     sigs = [125,125,125,125,125,125,95,95,95,125]
        for j in range(len(syn_types)):
            init_guess = [pmax[j], sigs, mu]
            r = minimize(
                fun=log_likelihood,
                x0=init_guess,
                bounds=[(0, 1.0), (40, 200), (0, 100)],
                method="nelder-mead",
                options={"maxfev": 1000},
                args=(syn_types[j], nonsyn_types[j]),
            )
            results.append(r)
    bins = np.array(range(0, upper_distance_limit, r_interval))
    bins = np.append(bins, r_interval + bins[-1])
    x = bins[1:] - (r_interval / 2)
    for i in range(len(unique_types)):
        pmodel = probfunct(results[i].x, syn_types[i])
        sorted_index = np.argsort(syn_types[i].r)
        sort_pm = pmodel[sorted_index]
        sort_synr = syn_types[i].r[sorted_index]

        sns.scatterplot(
            x=nonsyn_types[i].pt_position_x * (4 / 1000),
            y=nonsyn_types[i].pt_position_z * (40 / 1000),
            ax=ax[i, 0],
            color="grey",
            alpha=0.4,
            s=10,
        ).set_xlabel(r"$\mu$m (x vs z, top-down view)")
        sns.scatterplot(
            x=syn_types[i].pt_position_x * (4 / 1000),
            y=syn_types[i].pt_position_z * (40 / 1000),
            ax=ax[i, 0],
            size=syn_types[i].num_syn,
            hue=syn_types[i].num_syn,
            palette="crest",
            hue_norm=(0, 12),
            alpha=0.9,
            legend=False,
        )
        sns.scatterplot(
            x=pre.pt_position_x * (4 / 1000),
            y=pre.pt_position_z * (40 / 1000),
            marker="*",
            color="r",
            s=170,
            ax=ax[i, 0],
        ).set_ylabel(unique_types[i], fontsize=16)
        xrange = [
            int(pre.pt_position_x * (4 / 1000)) - 250,
            int(pre.pt_position_x * (4 / 1000)) + 250,
        ]
        yrange = [
            int(pre.pt_position_z * (40 / 1000)) - 250,
            int(pre.pt_position_z * (40 / 1000)) + 250,
        ]
        ax[i, 0].set_xlim(xrange[0], xrange[1])
        ax[i, 0].set_ylim(yrange[0], yrange[1])
        ax[i, 0].set_aspect("equal")

        method = "wilson"
        errorbars = sm.stats.proportion.proportion_confint(
            s_type[i], nobs=(s_type[i] + f_type[i]), method=method
        )
        probability = np.nan_to_num((s_type[i] / (f_type[i] + s_type[i])))
        ax[i, 1].scatter(
            x=x,
            y=probability,
            label="data:\n pmax={0:s}".format(str(pmax[i])),
            color="blue",
        )
        ax[i, 1].errorbar(
            x=x,
            y=probability,
            yerr=(probability - errorbars[0], errorbars[1] - probability),
            fmt="-o",
            alpha=0.6,
        )
        ax[i, 1].plot(
            sort_synr,
            sort_pm,
            color="r",
            ls="--",
            label="model:\n pmax={0:s}\n sigma={1:s}\n mu={2:s}".format(
                str(np.around(results[i].x[0], 4)),
                str(np.around(results[i].x[1], 1)),
                str(np.around(results[i].x[2], 1)),
            ),
        )
        ax[i, 1].set_xlabel(r"$\mu$m (radial)", fontsize=10)
        ax[i, 1].set_ylabel("Probability of Connection", fontsize=10)
        ax[0, 1].set_title("{0:s}".format(filename), fontsize=15, pad=20)
        ax[i, 1].legend()
        ax[i, 1].set_xlim(-1, (max(bins) + 5))
        ax[i, 1].set_ylim(-0.1, 1.0)
        ax[i, 1].grid()

        ax[i, 2].hist(
            x,
            bins=bins,
            weights=f_type[i] + s_type[i],
            density=False,
            label="Non-Synaptic",
            color="grey",
        )
        ax[i, 2].hist(
            x,
            bins=bins,
            weights=s_type[i],
            density=False,
            label="Synaptic",
            color="blue",
        )
        ax[i, 2].set_yscale("log")
        ax[i, 2].set_xlabel(r"$\mu$m (radial)", fontsize=10)
        ax[i, 2].set_xlim(-1, (max(bins) + 5))
        ax[i, 2].grid()
        ax[i, 2].set_ylabel("Log Frequency", fontsize=10)

    fig.tight_layout()
    if display == True:
        plt.show()
    else:
        plt.close(fig)
        import os

        if not os.path.exists(
            "./plots/{0:s}/{1:s}/".format(
                str(pre.cell_type.values[0]), str(upper_distance_limit)
            )
        ):
            os.makedirs(
                "./plots/{0:s}/{1:s}/".format(
                    str(pre.cell_type.values[0]), str(upper_distance_limit)
                )
            )
        fig.savefig(
            "./plots/{0:s}/{1:s}/{2:s}.pdf".format(
                str(pre.cell_type.values[0]), str(upper_distance_limit), filename
            )
        )


def makepdfs(
    client,
    pre,
    main,
    syn_types,
    nonsyn_types,
    s_type,
    f_type,
    r_interval,
    upper_distance_limit,
    MLEresults=True,
    threshold=None,
    display=False,
):
    if threshold == None:
        for i in tqdm(range(len(pre))):
            # this will give filename = '23_BC-123456-458760458604etc-15bin'
            try:
                preid = str(
                    client.materialize.query_table(
                        "allen_soma_coarse_cell_class_model_v1",
                        filter_equal_dict={"pt_root_id": pre[i].pt_root_id.values[0]},
                        select_columns=["id", "pt_root_id"],
                    ).id.values[0]
                )
                filename = "{0:s}-{1:s}-{2:s}-{3:s}bin".format(
                    str(np.array(pre[i].cell_type)[0]),
                    preid,
                    str(np.array(pre[i].pt_root_id)[0]),
                    str(r_interval),
                )
            except:
                filename = "{0:s}-{1:s}-{2:s}bin".format(
                    str(np.array(pre[i].cell_type)[0]),
                    str(np.array(pre[i].pt_root_id)[0]),
                    str(r_interval),
                )
            threepanels_pertype(
                main[i],
                pre[i],
                syn_types[i],
                nonsyn_types[i],
                s_type[i],
                f_type[i],
                r_interval,
                upper_distance_limit,
                filename,
                MLEresults,
                display,
            )
    else:
        for i in tqdm(range(len(pre))):
            # this will give filename = '23_BC-123456-4587604586etc-15bin-40synthresh'
            try:
                preid = str(
                    client.materialize.query_table(
                        "allen_soma_coarse_cell_class_model_v1",
                        filter_equal_dict={"pt_root_id": pre[i].pt_root_id.values[0]},
                        select_columns=["id", "pt_root_id"],
                    ).id.values[0]
                )
                filename = "{0:s}-{1:s}-{2:s}-{3:s}bin-{4:s}synthresh".format(
                    str(np.array(pre[i].cell_type)[0]),
                    preid,
                    str(np.array(pre[i].pt_root_id)[0]),
                    str(r_interval),
                    str(threshold),
                )
            except:
                filename = "{0:s}-{1:s}-{2:s}bin-{3:s}synthresh".format(
                    str(np.array(pre[i].cell_type)[0]),
                    str(np.array(pre[i].pt_root_id)[0]),
                    str(r_interval),
                    str(threshold),
                )
            threepanels_pertype(
                main[i],
                pre[i],
                syn_types[i],
                nonsyn_types[i],
                s_type[i],
                f_type[i],
                r_interval,
                upper_distance_limit,
                filename,
                MLEresults,
                display,
            )


def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    # Modified from https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html
    if not ax:
        ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[0, 0.1, 1], **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(row_labels)))
    ax.set_yticks(np.arange(len(col_labels)))
    ax.set_xticklabels(row_labels)
    ax.set_yticklabels(col_labels)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(len(row_labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(col_labels) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar


def annotate_heatmap(
    im,
    nconn,
    nprobe,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("white", "black"),
    threshold=None,
    **textkw
):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    # Normalize text color threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0
    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw1 = dict(horizontalalignment="center", verticalalignment="center", fontsize=13)
    kw1.update(textkw)
    kw2 = dict(horizontalalignment="center", verticalalignment="bottom", fontsize=8)
    kw3 = dict(horizontalalignment="center", verticalalignment="top", fontsize=8)
    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            try:
                kw1.update(color=textcolors[int(im.norm(data[i][j]) > threshold)])
                kw2.update(color=textcolors[int(im.norm(data[i][j]) > threshold)])
                kw3.update(color=textcolors[int(im.norm(data[i][j]) > threshold)])
                # if nconn[i][j] == 0:
                #     text1 = im.axes.text(j, i, valfmt(0, None), **kw1)
                # else:
                #     text1 = im.axes.text(j, i, valfmt(data[i][j], None), **kw1)
                text1 = im.axes.text(j, i, valfmt(data[i][j], None), **kw1)
                text2 = im.axes.text(j, i - 0.2, nconn[i][j], **kw2)
                text3 = im.axes.text(j, i + 0.2, nprobe[i][j], **kw3)
                texts.append(text1)
                texts.append(text2)
                texts.append(text3)
            except:
                kw1.update(color="grey")
                kw2.update(color="grey")
                kw3.update(color="grey")
                text1 = im.axes.text(j, i, data[i][j], **kw1)
                text2 = im.axes.text(j, i - 0.2, nconn[i][j], **kw2)
                text3 = im.axes.text(j, i + 0.2, nprobe[i][j], **kw3)
                texts.append(text1)
                texts.append(text2)
                texts.append(text3)
    return texts
