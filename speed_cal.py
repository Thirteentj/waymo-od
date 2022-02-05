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
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
def cal_speed_ave(trj_target_veh):
    speed_point = []
    for i in range(len(trj_target_veh)):
        v_x, v_y = trj_target_veh['velocity_x'].iloc[i], trj_target_veh['velocity_y'].iloc[i]
        v = math.sqrt(v_x ** 2 + v_y ** 2)
        speed_point.append(v)
    v_ave = np.mean(speed_point)
    return v_ave,speed_point


file_index = '00046'
filepath_trj = 'E:/Result_save/data_save/all_scenario_all_objects_info/' + file_index + '_all_scenario_all_object_info_1.csv'
seg_trj = pd.read_csv(filepath_trj)
single_seg_all_scenario_id = pd.unique(seg_trj['scenario_label'])
scenario_id = 22
turn_left_veh_id = 298
time_turn_begin = 2
time_turn_end = 15
trj_target_veh = seg_trj[(seg_trj['scenario_label']==scenario_id) & (seg_trj['obj_id']==turn_left_veh_id) & (seg_trj['valid']==True)]
print(trj_target_veh['time_stamp'])
trj_target_veh = trj_target_veh[(trj_target_veh['time_stamp']<=time_turn_end)&(trj_target_veh['time_stamp']>=time_turn_begin)]
v_ave,speed_point = cal_speed_ave(trj_target_veh)
print(v_ave)
