__all__  = ["threepanels_pertype","makepdfs"]

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels as sm
import warnings
from tqdm import tqdm

def threepanels_pertype(pre,syn_types,nonsyn_types,s_type,f_type,unique_types,r_interval,upper_distance_limit,filename):
    warnings.filterwarnings('ignore')

    fig, ax = plt.subplots(len(unique_types),3)
    fig.set_size_inches(12,26)

    bins = np.array(range(0,upper_distance_limit,r_interval))
    bins = np.append(bins,r_interval+bins[-1])
    x = bins[1:]-(r_interval/2)
    #x = x.clip(min=0)
    for i in range(len(unique_types)):
        sns.scatterplot(x=nonsyn_types[i].pt_position_x*(4/1000), y=nonsyn_types[i].pt_position_z*(40/1000),
                        ax=ax[i,0], color='grey', alpha=.4, s=10)
        sns.scatterplot(x=syn_types[i].pt_position_x*(4/1000), y=syn_types[i].pt_position_z*(40/1000),
                        ax=ax[i,0], color='b', alpha=.9, s=10).set_xlabel(r'$\mu$m (x vs z, top-down view)')
        sns.scatterplot(x=pre.pt_position_x*(4/1000), y=pre.pt_position_z*(40/1000), marker='*',color='r',s=200,
                        ax=ax[i,0]).set_ylabel(unique_types[i], fontsize=16)
        xrange = [int(pre.pt_position_x*(4/1000))-250,int(pre.pt_position_x*(4/1000))+250]
        yrange = [int(pre.pt_position_z*(40/1000))-250,int(pre.pt_position_z*(40/1000))+250]
        ax[i,0].set_xlim(xrange[0],xrange[1])
        ax[i,0].set_ylim(yrange[0],yrange[1])
        ax[i,0].set_aspect('equal')

        method = 'wilson'
        errorbars = sm.stats.proportion.proportion_confint(s_type[i],nobs=(s_type[i]+f_type[i]),method=method)
        probability = np.nan_to_num((s_type[i]/(f_type[i]+s_type[i])))
        ax[i,1].scatter(x=x, y=probability)
        ax[i,1].errorbar(x=x, y=probability, yerr=(probability-errorbars[0],errorbars[1]-probability), fmt='-o')
        ax[i,1].set_xlabel(r'$\mu$m (radial)', fontsize=10)
        ax[i,1].set_ylabel("Probability of Connection", fontsize=10)
        ax[i,1].set_ylim(-0.1,1.)
        ax[i,1].grid()

        ax[i,2].hist(x,bins=bins,weights=f_type[i]+s_type[i],density=False,label='Non-Synaptic', color='grey')
        ax[i,2].hist(x,bins=bins,weights=s_type[i],density=False,label='Synaptic', color='blue')
        ax[i,2].set_yscale('log')
        ax[i,2].set_xlabel(r'$\mu$m (radial)', fontsize=10)
        ax[i,2].set_xlim(-1,(max(bins)+5))
        ax[i,2].grid()
        ax[i,2].set_ylabel("Log Frequency", fontsize=10)

    fig.tight_layout()
    plt.close(fig)
    fig.savefig('./plots/{0:s}/{1:s}/{2:s}.pdf'.format(str(pre.cell_type.values[0]),str(upper_distance_limit),filename))

def makepdfs(client,pre,main,syn_types,nonsyn_types,s_type,f_type,r_interval,upper_distance_limit,threshold):
    if threshold == None:
        for i in tqdm(range(len(pre))):
            # this will give filename = 'BC-123456-458760458604etc-15bin'
            preid = str(client.materialize.query_table('allen_soma_coarse_cell_class_model_v1',
                                      filter_equal_dict = {'pt_root_id':pre[i].pt_root_id.values[0]},
                                      select_columns=['id','pt_root_id']).id.values[0])
            filename = '{0:s}-{1:s}-{2:s}-{3:s}bin'.format(str(np.array(pre[i].cell_type)[0]),preid,
            str(np.array(pre[i].pt_root_id)[0]),str(r_interval))
            unique_types = np.unique(main[i].cell_type)
            threepanels_pertype(pre[i],syn_types[i],nonsyn_types[i],s_type[i],f_type[i],
                                unique_types,r_interval,upper_distance_limit,filename)
    else:
        for i in tqdm(range(len(pre))):
            # this will give filename = 'BC-123456-4587604586etc-15bin-40synthresh'
            preid = str(client.materialize.query_table('allen_soma_coarse_cell_class_model_v1',
                                      filter_equal_dict = {'pt_root_id':pre[i].pt_root_id.values[0]},
                                      select_columns=['id','pt_root_id']).id.values[0])
            filename = '{0:s}-{1:s}-{2:s}-{3:s}bin-{4:s}synthresh'.format(str(np.array(pre[i].cell_type)[0]),preid,
            str(np.array(pre[i].pt_root_id)[0]),str(r_interval),str(threshold))
            unique_types = np.unique(main[i].cell_type)
            threepanels_pertype(pre[i],syn_types[i],nonsyn_types[i],s_type[i],f_type[i],
                                unique_types,r_interval,upper_distance_limit,filename)