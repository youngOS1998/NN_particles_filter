from random import shuffle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import torch 
import torch.nn as nn

class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, Flag_train=True):
        self.Flag = Flag_train
        if self.Flag:
            self.path_training_data = './data_particles_training/training_data_new_3.csv'
        else:
            self.path_training_data = './data_particles_training/evaluation_data_new_3.csv'
        self.picture = 'map1.png'
        self.size = (20, 35)

        self.data = self._get_training_data()   # 数据是（len_data，8）

    def __getitem__(self,idx): 
        return self.data[idx]

    def __len__(self): # What is the length of the dataset
        return len(self.data)

    def _get_map(self):
        gray_1 = cv2.imread(self.picture, cv2.IMREAD_GRAYSCALE)
        print(type(gray_1))
        # size = (20, 35)  # 20是宽， 35是高
        gray = cv2.resize(gray_1, dsize=self.size, interpolation=cv2.INTER_AREA)
        gray_binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        # m, n = gray.shape
        # for i in range(m):
        #     for j in range(n):
        #         if gray[i,j] > 200:
        #             gray[i,j] = 255
        #         else:
        #             gray[i,j] = 0

        # print('gray shape is:', gray.shape)
        # fig  = plt.figure()
        # ax   = fig.gca(projection='3d')
        # X    = np.arange(0, gray.shape[1], 1)
        # Y    = np.arange(0, gray.shape[0], 1)
        # X, Y = np.meshgrid(X, Y)
        # R    = gray
        # surf = ax.plot_surface(X, Y, R, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # plt.show()
        return gray_binary                                         # gray.shape = (35, 20)  即35行，20列

    def maxminnorm(self, array):                            # 用于做归一化，到0~1之间
        maxcols=array.max(axis=0)
        mincols=array.min(axis=0)
        data_shape = array.shape
        data_rows = data_shape[0]
        data_cols = data_shape[1]
        t=np.empty((data_rows,data_cols))
        for i in range(data_cols):
            t[:,i]=(array[:,i]-mincols[i])/(maxcols[i]-mincols[i])
        return t
        
    def _get_training_data(self):
        raw_data = pd.read_csv(self.path_training_data)     # 此数据格式为：'x', 'y', 'dist_up', 'dist_right', 'dist_bottom', 'dist_left', 'dist_around'
                                                            # 我们要将其(7)和图片(20x35)组合在一起，合成一个数据送入神经网络中
        # 周围随机点的坐标
        if not self.Flag:
            dist_index_sort = np.array(raw_data['dist_index_sort'])
            dist_index_sort = dist_index_sort[:, np.newaxis]

        random_up        = np.array(raw_data['random_up'     ])
        random_right     = np.array(raw_data['random_right'  ])
        random_bottom    = np.array(raw_data['random_bottom' ])
        random_left      = np.array(raw_data['random_left'   ])
        # 中间点的坐标
        center_up        = np.array(raw_data['center_up'     ])
        center_right     = np.array(raw_data['center_right'  ])
        center_bottom    = np.array(raw_data['center_bottom' ])
        center_left      = np.array(raw_data['center_left'   ])
        # labels
        dist             = np.array(raw_data['dist_around'   ]) 


        random_up        = random_up[:, np.newaxis]                      # shape = len_data x 1
        random_right     = random_right[:, np.newaxis]
        random_bottom    = random_bottom[:, np.newaxis]
        random_left      = random_left[:, np.newaxis]
        center_up        = center_up[:, np.newaxis]
        center_right     = center_right[:, np.newaxis]
        center_bottom    = center_bottom[:, np.newaxis]
        center_left      = center_left[:, np.newaxis]
        dist             = dist[:, np.newaxis]
        dist             = self.maxminnorm(dist)

        len_data         = len(random_up)         # 数据的长度

        # # 接下来传入图片的数据
        # picture_data = self._get_map()   # (35, 20)
        # pic_reshaped = picture_data.reshape((1,-1))
        # pic_reshaped_repeat = np.repeat(pic_reshaped, len_data, axis=0)
        # concate_data = np.concatenate((pic_reshaped_repeat, x, y, dist_up, dist_right, dist_bottom, dist_left, dist_around), axis=1)  # shape = (len_data, 707)
        if self.Flag:
            concate_data = np.concatenate((random_up, random_right, random_bottom, random_left, center_up, center_right, center_bottom, \
                                        center_left, dist), axis=1)   # shape = 9    data = 9  label = 1
        else:
            concate_data = np.concatenate((random_up, random_right, random_bottom, random_left, center_up, center_right, center_bottom, \
                            center_left, dist, dist_index_sort), axis=1)   # shape = 10   
        return concate_data