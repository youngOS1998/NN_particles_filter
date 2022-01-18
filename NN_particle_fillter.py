from random import shuffle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import torch 
import torch.nn as nn


class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.path_training_data = './data_particles_training/training_data.csv'
        self.picture = 'map1.png'

        self.data = self._get_training_data()   # 数据是（len_data，707）

    def __getitem__(self,idx): 
        return self.data[idx]

    def __len__(self): # What is the length of the dataset
        return len(self.data)

    def _get_map(self):
        gray_1 = cv2.imread(self.picture, cv2.IMREAD_GRAYSCALE)
        print(type(gray_1))
        # size = (20, 35)  # 20是宽， 35是高
        gray = cv2.resize(gray_1, dsize=self.size, interpolation=cv2.INTER_AREA)
        m, n = gray.shape
        for i in range(m):
            for j in range(n):
                if gray[i,j] > 200:
                    gray[i,j] = 255
                else:
                    gray[i,j] = 0

        print('gray shape is:', gray.shape)
        fig  = plt.figure()
        ax   = fig.gca(projection='3d')
        X    = np.arange(0, gray.shape[1], 1)
        Y    = np.arange(0, gray.shape[0], 1)
        X, Y = np.meshgrid(X, Y)
        R    = gray
        surf = ax.plot_surface(X, Y, R, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.show()
        return gray     # gray.shape = (35, 20)  即35行，20列
        
    def _get_training_data(self):
        raw_data = pd.read_csv(self.path_training_data)     # 此数据格式为：'x', 'y', 'dist_up', 'dist_right', 'dist_bottom', 'dist_left', 'dist_around'
                                                            # 我们要将其(7)和图片(20x35)组合在一起，合成一个数据送入神经网络中
        x           = np.array(raw_data['x'          ])
        y           = np.array(raw_data['y'          ])
        dist_up     = np.array(raw_data['dist_up'    ])
        dist_right  = np.array(raw_data['dist_right' ])
        dist_bottom = np.array(raw_data['dist_bottom'])
        dist_left   = np.array(raw_data['dist_left'  ])
        dist_around = np.array(raw_data['dist_around'])

        x           = x[:, np.newaxis]                      # shape = len_data x 1
        y           = y[:, np.newaxis]
        dist_up     = dist_up[:, np.newaxis]
        dist_right  = dist_right[:, np.newaxis]
        dist_bottom = dist_bottom[:, np.newaxis]
        dist_left   = dist_left[:, np.newaxis]
        dist_around = dist_around[:, np.newaxis]

        len_data    = len(x)         # 数据的长度

        # 接下来传入图片的数据
        picture_data = self._get_map()   # (35, 20)
        pic_reshaped = picture_data.reshape((1,-1))
        pic_reshaped_repeat = np.repeat(pic_reshaped, len_data, axis=0)
        concate_data = np.concatenate((pic_reshaped_repeat, x, y, dist_up, dist_right, dist_bottom, dist_left, dist_around), axis=1)  # shape = (len_data, 707)
        return concate_data



class NN_particle_filter(nn.Model):

    def __init__(self, num_classes=1, init_weights=False):
        super(NN_particle_filter, self).__init__()
        # define our neuron network below

        self.net = nn.Sequential(
            nn.Linear(707,  6400), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096,   10), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(10,      1)
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):   # 若是卷积层
                nn.init.normal_(m.weight, mean=0, std=1)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear): # 若是全连接层
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0) 

        
if __name__ == '__main__':

    network = NN_particle_filter(init_weights=True)
    use_cuda = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0002)

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    dataset_1 = ExampleDataset()
    dataLoader = torch.utils.data.DataLoader(dataset=dataset_1, shuffle=True, batch_size=10)   # 每次送入的数据是10 x 707, 10指的是batch size, 
                                                                                               # 707指的是数据的维度

    for epoch in range(10):
        network.train()
        running_loss = 0.0

        for step, datapoint in enumerate(dataLoader):   # datapoint 的shape是(10, 707)
            data, label = datapoint[:, 0:706], datapoint[:, 706]
            save_path   = './AlexNet.pth'
            beat_acc    = 0.0
            
            i = 0
            loss_list = []
