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

def get_file_list(filepath):
    all_files = sorted(glob.glob(filepath))
    segs_name_index = []
    for file in all_files:
        segment_name = os.path.basename(file)
        segs_name_index.append(segment_name[-14:-9])
    # print(segs_name_all)
    #print(segs_name_index)
    return all_files, segs_name_index

def map_lane_id_all(map_features,file_index,scenario_label):
    line_id_all = []  #所有线段ID信息，包括车道线、边界线等
    lane_id_all = []  #所有车道的ID信息
    for single_feature in map_features:
        line_id_all.append(single_feature.id)
        if list(single_feature.lane.polyline) != []:
            lane_id_all.append(single_feature.id)
    line_num = len(line_id_all)
    lane_num = len(lane_id_all)
    line_id_ave = np.mean(line_id_all)
    lane_id_ave = np.mean(lane_id_all)
    dic = {}
    dic['file_index'] = file_index
    dic['scenario_label'] = scenario_label
    dic['line_id_ave'] = line_id_ave
    dic['line_num'] = line_num
    dic['lane_id_ave'] = lane_id_ave
    dic['lane_num'] = lane_num
    return dic


if __name__ == '__main__':
    #该脚本用于检查是否存在重复场景，但是目前效果不理想，需要继续调整
    #-----------------------load_data ---------------------
    test_state = 0
    filepath_oridata = 'E:/waymo_motion_dataset/training_20s.tfrecord-*-of-01000'
    all_file_list, file_index_list = get_file_list(filepath_oridata)
    filepath_turn_left_scenario = 'E:/Result_save/data_save//Turn_left_scenario_all.csv'
    df_turn_left_scenario = pd.read_csv(filepath_turn_left_scenario)
    scenario_all_count = 0
    # all_intersection_info = []
    # length = 80   #目标交叉口的距离范围，需要进行调整
    df_all_seg_id = pd.DataFrame()
    for i in tqdm(range(len(file_index_list))):
        single_segment_all_scenario_all_lane_topo_info = []
        file_index = file_index_list[i]
        segment_file = all_file_list[i]
        print('Now is the file:%s' % file_index)
        # filepath_trj = 'data_save/all_scenario_all_objects_info/'+ file_index +'_all_scenario_all_object_info_1.csv'
        # seg_trj = pd.read_csv(filepath_trj)
        # single_seg_all_scenario_id = pd.unique(seg_trj['scenario_label'])
        segment_name_list = []
        segment_dataset = tf.data.TFRecordDataset(segment_file)
        segment_dataset = segment_dataset.apply(tf.data.experimental.ignore_errors())
        
        if test_state ==1:
            if i >50:
                break
        scenario_label = 0  # 所有场景的ID数量记录
        df_single_seg_all_scenario_lane_info = pd.DataFrame()
        single_seg_all_scenario_lane_info = []
        #single_file_all_scenario = []
        for one_scenario in segment_dataset:  # one_scenario 就是一个scenario
            single_scenario_all_feature = []
            scenario_label += 1
            scenario_all_count += 1
            #print('Now is the scenario:%s' % scenario_label)
            if test_state == 1:
                target_scenario = 18
                #intersection_center_loc = (1798.4981814285616, -2136.2457659217735)
                # if scenario_label != target_scenario:
                #     continue
                if scenario_label > 50:
                    break
            scenario = Scenario()
            scenario.ParseFromString(one_scenario.numpy())  # 数据格式转化
            time_stamp_all = scenario.timestamps_seconds
            map_features = scenario.map_features

            # output = "D:/Data/WaymoData/" + "scenario_" + str(scenario_label) + ".txt"
            # data = open(output, 'w+')
            # print(scenario, file=data)
            # data.close()

            #line_id_ave, line_num, lane_id_ave, lane_num = map_lane_id_all(map_features)
            if file_index == '00000':
                segment_id = 0
            else:
                segment_id = eval(file_index.strip('0'))
            scenario_label_left_list = df_turn_left_scenario[(df_turn_left_scenario['segment_id']==segment_id)]['scenario_id'].to_list()
            #print(scenario_label_left_list)
            if scenario_label in scenario_label_left_list:
                dic = map_lane_id_all(map_features,file_index,scenario_label)
                single_seg_all_scenario_lane_info.append(dic)

        df_single_seg_all_scenario_lane_info = pd.DataFrame(single_seg_all_scenario_lane_info)
        df_all_seg_id = pd.concat([df_all_seg_id,df_single_seg_all_scenario_lane_info])




            #print('road_edge_count {},lane_count {},road_line {},all_count {}'.format(road_edge_count, lane_count, road_line,all_element_count))

    #-----------------ouput------------------
    outpath_lane_id_info = 'E:/Result_save/data_save/left_all_seg_all_lane_id_info' + '.csv'  #现在是一个scenario导出一个csv,后面需要导出一个完整的
    #print(outpath_lane_id_info)

    df_all_seg_id.to_csv(outpath_lane_id_info)
    print('Result has been printed')