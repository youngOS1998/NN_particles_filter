#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
import os
from nav_msgs.msg import Odometry

class motionDetector:
    def __init__(self):
        rospy.on_shutdown(self.cleanup)

        # 创建cv_bridge
        self.map_path = './map/maze_map.png'
        self.map = self.read_map(self.map_path)   # 读取的map

        # 初始化订阅rgb格式图像数据的订阅者，此处图像topic的话题名可以在launch文件中重映射
        self.image_sub = rospy.Subscriber("/robot0/odom", Odometry, self.image_callback, queue_size=1)

    def read_map(self, file_pathname):  
        print(file_pathname)
        img = cv2.imread(file_pathname)
        return img

    def plot_position(self, img, x, y):
        center = (x, y)
        circle_color = (255, 0, 0)             # 此圆为红色
        circle_radius = 10                     # 圆的半径
        circle_size = 1                        # 线的宽度

        cv2.circle(img, center, circle_radius, circle_color, circle_size)   # 画一个圆表示机器人当前的位置
        cv2.imshow(img)
        cv2.waitKey(0)



    def image_callback(self, data):
        x, y = data.pose.pose.position.x, data.pose.pose.position.y         # 此为机器人当前的位置
        print('the position of robot is: ({0}, {1})'.format(x, y))
        self.plot_position(self.map, x, y)

    def cleanup(self):
        print("Shutting down vision node.")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("motion_detector")
        rospy.loginfo("motion_detector node is started...")
        rospy.loginfo("Please subscribe the ROS image.")
        motionDetector()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down motion detector node.")
        cv2.destroyAllWindows()