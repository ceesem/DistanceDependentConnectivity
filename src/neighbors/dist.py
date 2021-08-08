__all__ = ["Euc_cell2cell","Rad_cell2cell","Euc_syn2cell","Rad_syn2cell"]

import numpy as np

def Euc_cell2cell(pre,post):
    # adjusts coordinates to be in units of microns
    xy = (4./1000)
    z = (40./1000)
    x_pre,y_pre,z_pre = pre.pt_position_x.values*xy,pre.pt_position_y.values*xy,pre.pt_position_z.values*z
    x_pos,y_pos,z_pos = post.pt_position_x.values*xy,post.pt_position_y.values*xy,post.pt_position_z.values*z
    d = np.zeros(len(post))
    for i in range(len(post)):
        d[i] = np.around(np.sqrt((x_pre-x_pos[i])**2 + (y_pre-y_pos[i])**2 + (z_pre-z_pos[i])**2),3)
    return d

def Rad_cell2cell(pre,post):
    # adjusts coordinates to be in units of microns
    xy = (4./1000)
    z = (40./1000)
    x_pre,z_pre = pre.pt_position_x.values*xy,pre.pt_position_z.values*z
    x_pos,z_pos = post.pt_position_x.values*xy,post.pt_position_z.values*z
    d = np.zeros(len(post))
    for i in range(len(post)):
        d[i] = np.around(np.sqrt((x_pre-x_pos[i])**2 + (z_pre-z_pos[i])**2),3)
    return d


def Euc_syn2cell(syn_df,cell):
    # adjusts coordinates to be in units of microns
    xy = (4./1000)
    z = (40./1000)
    x_cell,y_cell,z_cell = cell.pt_position_x.values*xy,cell.pt_position_y.values*xy,cell.pt_position_z.values*z
    distance = []
    for i in range(len(syn_df)):
        d = []
        if syn_df.ctr_pt_position[i] == 0:
            d.append(0)
        else:
            for j in range(len(np.array(syn_df.ctr_pt_position[i]))):
                x_syn = np.array(syn_df.ctr_pt_position[i])[j][0]*xy
                y_syn = np.array(syn_df.ctr_pt_position[i])[j][1]*xy
                z_syn = np.array(syn_df.ctr_pt_position[i])[j][2]*z
                if len(cell) == 1:
                    dsyn = np.sqrt((x_syn-x_cell[0])**2 + (y_syn-y_cell[0])**2 + (z_syn-z_cell[0])**2)
                else:
                    dsyn = np.sqrt((x_syn-x_cell[i])**2 + (y_syn-y_cell[i])**2 + (z_syn-z_cell[i])**2)
                d.append(np.around(dsyn,3))
        distance.append(d)
    return distance

def Rad_syn2cell(syn_df,cell):
    # adjusts coordinates to be in units of microns
    xy = (4./1000)
    z = (40./1000)
    x_cell,z_cell = cell.pt_position_x.values*xy,cell.pt_position_z.values*z
    distance = []
    for i in range(len(syn_df)):
        d = []
        if syn_df.ctr_pt_position[i] == 0:
            d.append(0)
        else:
            for j in range(len(np.array(syn_df.ctr_pt_position[i]))):
                x_syn = np.array(syn_df.ctr_pt_position[i])[j][0]*xy
                z_syn = np.array(syn_df.ctr_pt_position[i])[j][2]*z
                if len(cell) == 1:
                    dsyn = np.sqrt((x_syn-x_cell[0])**2 + (z_syn-z_cell[0])**2)
                else:
                    dsyn = np.sqrt((x_syn-x_cell[i])**2 + (z_syn-z_cell[i])**2)
                d.append(np.around(dsyn,3))
        distance.append(d)
    return distance