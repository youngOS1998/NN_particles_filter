import numpy as np
import pandas as pd
import cv2
import random
import math

class create_training_data():
    def __init__(self) -> None:
        self.points_num = 100
        self.size = (20,35)
        self.picture = 'map1.png'
        self.map = self.get_map()
        # self.columns_name = {'x', 'y', 'dist_up', 'dist_right', 'dist_bottom', 'dist_left', 'dist_around'}
        # self.data_Frame = pd.DataFrame(columns=self.columns_name)


    def get_map(self):
        gray_1 = cv2.imread(self.picture, cv2.IMREAD_GRAYSCALE)
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
        return gray
    

    def dist_to_round(self, i, j, direction):            # 从i, j这个点到周围上下左右的距离
        if direction == 'up':
            temp_up = i
            while (temp_up - 1) >= 0 and self.map[temp_up - 1][j] != 0:
                temp_up = temp_up - 1
            dist_up = abs(i - temp_up)
            return dist_up
        if direction == 'right':
            temp_right = j
            while ((temp_right + 1) <= (self.size[0] - 1)) and self.map[i][temp_right + 1] != 0:
                temp_right = temp_right + 1
            dist_right = abs(j - temp_right)
            return dist_right
        if direction == 'bottom':
            temp_bottom = i
            while ((temp_bottom + 1) <= (self.size[1] - 1)) and self.map[temp_bottom + 1][j] != 0:
                temp_bottom = temp_bottom + 1
            dist_bottom = abs(i - temp_bottom)
            return dist_bottom
        if direction == 'left':
            temp_left = j
            while ((temp_left - 1) >= 0) and self.map[temp_left - 1][j] != 0:
                temp_left = temp_left - 1
            dist_left = abs(j - temp_left)
            return dist_left


    def create_data(self):
        m, n = self.map.shape                 # 输入地图的尺寸

        columns_name = {'x', 'y', 'dist_up', 'dist_right', 'dist_bottom', 'dist_left', 'dist_around'}
        data_Frame = pd.DataFrame(columns=columns_name)

        for i in range(self.points_num):      # 每次迭代在这个尺寸地图中随机产生一个点, 然后在整个地图中产生100个随机点，形成训练数据。
            point_center_x = random.randint(0, m-1)   # 产生中心点的行
            point_center_y = random.randint(0, n-1)   # 产生中心点的列
            
            # 我们让随机点到中心点的距离的反比 (1/distance) 成为这个点的真实权重，即训练时的label
            for j in range(100):              # 围绕这个中心点在整个地图中产生100个随机点
                data_single_x = random.randint(0, m-1)
                data_single_y = random.randint(0, n-1)
                if self.map[data_single_x][data_single_y] != 0:
                    dist = math.sqrt(math.pow((data_single_x - point_center_x), 2) + math.pow((data_single_y - point_center_y), 2))  # 周围随机点到中心点的距离
                    dist_around = round(dist, 3)
                    dist_up = self.dist_to_round(data_single_x, data_single_y, 'up')
                    dist_right = self.dist_to_round(data_single_x, data_single_y, 'right')
                    dist_bottom = self.dist_to_round(data_single_x, data_single_y, 'bottom')
                    dist_left = self.dist_to_round(data_single_x, data_single_y, 'left')
                    dist_info = {'x':[data_single_x], 'y':[data_single_y], 'dist_up':[dist_up], 'dist_right'\
                                :[dist_right], 'dist_bottom':[dist_bottom], 'dist_left':[dist_left], 'dist_around':[dist_around]}  # 每个单独点的信息
                    new_frame = pd.DataFrame(dist_info)
                    data_Frame = data_Frame.append(new_frame, ignore_index=True)
                    print('---center point---: {0},  ---surrounding points---{1}'.format(i, j))
                    print('x:{0}, y:{1}, up:{2}, right:{3}, bottom:{4}, left:{5}\n'.format(data_single_x, data_single_y, dist_up, dist_right, dist_bottom, dist_left))
        return data_Frame

    def save_data(self, data, path='./data_particles_training/training_data.csv'):
        print('save data!')
        data.to_csv(path)
                
if __name__ == '__main__':

    create_data = create_training_data()
    training_data = create_data.create_data()
    create_data.save_data(training_data)


                
                
