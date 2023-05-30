import numpy as np

def euclidean_distance(xyz0, xyz):
    xyz = np.atleast_2d(xyz)
    return np.sqrt(
        ((xyz0[0] - xyz[:,0]) ** 2) + ((xyz0[1] - xyz[:,1]) ** 2) + ((xyz0[2] - xyz[:,2]) ** 2)
    )

def radial_distance(xyz0, xyz):
    xyz = np.atleast_2d(xyz)
    return np.sqrt(
        ((xyz0[0] - xyz[:,0]) ** 2) + ((xyz0[2] - xyz[:,2]) ** 2)
    )