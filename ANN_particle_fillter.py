import heapq
from random import shuffle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import torch 
import torch.nn as nn
from Dataset import ExampleDataset
import time


class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
    
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NN_particle_filter(nn.Module):
    """
    这个类的作用是生成一个ANN网络对周围粒子的重要度进行估计
    """

    def __init__(self, num_classes=1, init_weights=False):
        super(NN_particle_filter, self).__init__()
        # define our neuron network below

        self.net_particles = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):                    # around_x是机器人周围粒子的特征
        """前向传播"""
        return self.net_particles(x)         # 此是从图片中提取出的特征
    
    def loss_eval(self):
        """在测试时得到的"""
        pass


    def _initialize_weights(self):
        """给网络层赋予初始权重和偏差"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):   # 若是卷积层
                nn.init.normal_(m.weight, mean=0, std=1)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear): # 若是全连接层
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0) 
    

    def accuracy(self, y_hat, y):
        """计算预测正确的数量"""
        cmp = y_hat == y
        return float(cmp.sum())
        

    # def evaluate_accuracy(self, net, data_iter):  # training: data_iter:  9 x 1     testing: data_iter: 10 x 1 
    #     """验证时，计算在指定数据集上模型的精度"""
    #     if isinstance(net, torch.nn.Module):
    #         net.eval()
    #     metric = Accumulator(2)
    #     for step, data_point in enumerate(data_iter):
    #         X, y = data_point[:, 0:8], data_point[:, 9]    # X shape: 100 x 8,  y shape: 100 x 1
    #         output = self.forward(X.float())                       # output shape: 100 x 1

    #         index_s = heapq.nsmallest(100, output)
    #         index_sort = map(index_s.index, index_s)
    #         metric.add(self.accuracy(index_sort, y), y.numel())
        
    #     return metric[0] / metric[1] 

        
if __name__ == '__main__':

    network  = NN_particle_filter(init_weights=True)
    use_cuda = True
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    loss_function = nn.MSELoss()
    optimizer     = torch.optim.SGD(network.parameters(), lr=0.002)

    kwargs        = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    dataset_1     = ExampleDataset()
    dataLoader    = torch.utils.data.DataLoader(dataset=dataset_1, shuffle=True, batch_size=30)   # 每次送入的数据是10 x 707, 10指的是batch size, 
                                                                                               # 707指的是数据的维度
    i = 0
    loss_list     = []                                                                                           
    save_path     = './NN_particles_filter.pth'
    beat_acc      = 0.0

    for epoch in range(10):   # 对全部样本进行30次遍历
        network.train()
        running_loss = 0.0

        for step, datapoint in enumerate(dataLoader):              # datapoint 的shape是(10, 707)
            data, label = datapoint[:, 0:8], datapoint[:, 8]
            data = data.float()
            label = label.float()
            optimizer.zero_grad()                                  # 清除历史梯度
            outputs = network.forward(data.to(device))
            print(outputs)
            time.sleep(4)

            loss = loss_function(outputs, label.to(device))
            loss.backward()
            loss_copy = loss.detach()
            optimizer.step()

            if i % 10 == 0:
                loss_list.append(loss_copy)

            rate = (step + 1) / len(dataLoader)
            a = '*' * int(rate * 50)
            b = '.' * int((1 - rate) * 50)

            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
            i = i + 1

    torch.save(network.state_dict(), 'parameter_3.pkl')

    loss_save = np.array(loss_list)
    np.save('./loss_save_2.npy', loss_save)
            

