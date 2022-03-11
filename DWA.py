from cmath import pi
import numpy as np
import random

class DWA_model():

    def __init__(self) -> None:
        pass

    def get_Obstacle(self):                        # 得到障碍物
        obstacle = np.array([[3, 6, 8, 2, 4, 7, 9], [10*random.random, 10*random.random(), 10*random.random(), 5, 2, 7, 9]])
        for 


    def DrawObstacle(self, obstacle, obstacleR):   # 绘制所有障碍物的位置, obstacle: 所有障碍物的坐标
        r = obstacleR   # 障碍物半径
        theta = np.arange(0, 2*np.pi, np.pi/20)
        for id in range():