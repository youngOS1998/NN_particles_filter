from random import shuffle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import torch 
import torch.nn as nn
from ANN_particle_fillter import NN_particle_filter, Accumulator
import heapq
from Dataset import ExampleDataset

# def evaluate_accuracy(net, data_iter):  # training: data_iter:  9 x 1     testing: data_iter: 10 x 1 
#     """验证时，计算在指定数据集上模型的精度"""
#     if isinstance(net, torch.nn.Module):
#         net.eval()
#     metric = Accumulator(2)
#     for step, data_point in enumerate(data_iter):
#         X, y = data_point[:, 0:8], data_point[:, 9]    # X shape: 100 x 8,  y shape: 100 x 1
#         output = net.forward(X.float())                       # output shape: 100 x 1
#         index_s = heapq.nsmallest(100, output)
#         index_sort = map(index_s.index, index_s)
#         metric.add(net.accuracy(index_sort, y), y.numel())
    
#     return metric[0] / metric[1]

def evaluate_accuracy(net, data_iter):  # training: data_iter:  9 x 1     testing: data_iter: 10 x 1 
    """验证时，计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    predict_list = []
    columns_name = {'output'}
    data_Frame   = pd.DataFrame(columns=columns_name)
    
    for step, data_point in enumerate(data_iter):
        X, y = data_point[:, 0:8], data_point[:, 9]    # X shape: 100 x 8,  y shape: 100 x 1
        print(X)
        output = net.forward(X.float())                       # output shape: 100 x 1
        print(output)
        dist_info   = {'output':[output.detach()]}
        new_frame   = pd.DataFrame(dist_info)
        data_Frame.append(new_frame, ignore_index=True)
    return data_Frame


if __name__ == '__main__':

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_eval = NN_particle_filter()
    model_eval.eval()
    model_eval.load_state_dict(torch.load('parameter_2.pkl'))

    use_cuda = True
    kwargs = {'num_workers':0, 'pin_memory':True} if use_cuda else {}
    loss_function = nn.MSELoss()
    dataset_eval = ExampleDataset(Flag_train=False)
    test_loader    = torch.utils.data.DataLoader(dataset=dataset_eval, shuffle=False, batch_size=100)   # 此时在测试时我们不需要打乱（shuffle）
                                                                                                        # 且此处batch_size=100, 表明此处把一个点
                                                                                                        # 周围的所有点都取出了
    i = 0

    loss_list = []
    not_equal_num = 0
    all_num = len(test_loader)

    for epoch in range(1): 
        running_loss = 0.0
        data = evaluate_accuracy(model_eval, test_loader)
        data.to_csv('output_1.csv')
        # print(acc)

    # loss_save = np.array(loss_list)
    # np.save('./loss_eval_save.npy', loss_save)
