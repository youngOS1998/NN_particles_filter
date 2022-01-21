import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('./data_particles_training/training_data.csv')
dist_x = np.array(data['x'])
dist_y = np.array(data['y'])
# plt.scatter(dist_x, dist_y)
# plt.show()

def maxminnorm(array):             
    maxcols=array.max(axis=0)
    mincols=array.min(axis=0)
    data_shape = array.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t=np.empty((data_rows,data_cols))
    for i in range(data_cols):
        t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
    return t

dist_around = np.array(data['dist_around'])
dist_around = dist_around[:, np.newaxis]
temp = maxminnorm(dist_around)
print(temp.shape)