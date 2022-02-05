import os
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from waymo_open_dataset.protos.scenario_pb2 import Scenario

def get_lane_passing(veh_1,veh_2):  #得到两辆车整个过程所有车道的ID信息
    lane_list_left = []  #左转车所经过的所有车道
    lane_list_straight = [] #直行车所经过的所有车道


    return lane_list_left,lane_list_straight

def get_conflict_point(lane_list_left,lane_list_straight,all_lane_topo_info,single_map_dict):  #all_lane_topo_info 是该scenario下所有的车道信息

    lane_left_id = -1  #左转车道
    lane_straight_id = -1  #直行车辆经过的交叉口内部的直行车道
    for lane in lane_list_left:
        if all_lane_topo_info[all_lane_topo_info['lane_id']==lane]['is_a_turn'] == 'left':
            lane_left_id = lane
            break
    for lane in lane_list_straight:
        if (all_lane_topo_info[all_lane_topo_info['lane_id']==lane]['is_a_turn'] == 'straight') and (all_lane_topo_info[all_lane_topo_info['lane_id']==lane]['entry_or_exit'] == 'inside'):
            lane_straight_id = lane
            break

    lane_left_polyline = single_map_dict[lane_left_id]
    lane_straight_polyline = single_map_dict[lane_straight_id]

    #计算两根车道的交点
    min_dis = 9999
    min_dis_point = ()
    intersection_point = ()
    for lane_left_point in lane_left_polyline:
        x1,y1 = lane_left_point.x,lane_left_point.y
        for lane_straight_point in lane_straight_polyline:
            x2,y2 = lane_straight_point.x,lane_straight_point.y

            dis = math.sqrt((x2-x1)**2+(y2-y1)**2)
            if dis <= min_dis:
                min_dis = dis
                min_dis_point = (x2,y2)
    if min_dis <=0.01:
        intersection_point = min_dis_point

    return intersection_point

def cal_PET(intersection_point,seg_trj,veh_left,veh_straight):   #这里的seg_trj是目标scenario的trj
    target_point_x = intersection_point[0]
    target_point_y = intersection_point[1]
    veh_left_trj = seg_trj[(seg_trj['obj_id']==veh_left) & ((target_point_x-3)<=seg_trj['center_x']<=(target_point_x+3)) & ((target_point_y-3)<=seg_trj['center_y']<=(target_point_y+3))]  #初步筛选目标车辆轨迹点
    veh_straight_trj = seg_trj[(seg_trj['obj_id']==veh_straight) &((target_point_x-3)<=seg_trj['center_x']<=(target_point_x+3)) & ((target_point_y-3)<=seg_trj['center_y']<=(target_point_y+3)) ]

    min_dis_left = 9999 #记录冲突点所在位置
    time_stamp_min_dis_left = -1  #左转车辆通过冲突点时的时间戳
    for i in range(len(veh_left_trj)):
        x_veh,y_veh = veh_left_trj.iloc[i].center_x,veh_left_trj.iloc[i].center_y
        time_stamp_left = veh_left_trj.iloc[i].time_stamp
        dis = math.sqrt((x_veh-target_point_x)**2+(y_veh-target_point_y)**2)
        if dis<min_dis_left:
            dis = min_dis_left
            time_stamp_min_dis_left = time_stamp_left

    min_dis_straight = 9999  # 记录冲突点所在位置
    time_stamp_min_dis_straight = -1  # 左转车辆通过冲突点时的时间戳
    for i in range(len(veh_straight_trj)):
        x_veh, y_veh = veh_straight_trj.iloc[i].center_x, veh_straight_trj.iloc[i].center_y
        time_stamp_straight = veh_straight_trj.iloc[i].time_stamp
        dis = math.sqrt((x_veh - target_point_x) ** 2 + (y_veh - target_point_y) ** 2)
        if dis < min_dis_straight:
            dis = min_dis_straight
            time_stamp_min_dis_straight = time_stamp_straight

    PET = time_stamp_min_dis_straight-time_stamp_min_dis_left  #如果大于0表明直行先于左转通过，小于0说明左转先于直行通过

    return PET









