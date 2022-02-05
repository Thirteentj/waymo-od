import os
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from waymo_open_dataset.protos.scenario_pb2 import Scenario
import Extract01 as Ext


if __name__ == '__main__':
    #-----------------------load_data ---------------------

    test_state = 0
    filepath_oridata = 'D:/Data/WaymoData_motion_1/training_20s.tfrecord-*-of-01000'
    all_file_list, file_index_list = Ext.get_file_list(filepath_oridata)
    filepath_turn_left_scenario = 'D:/Data/Git/waymo-od/data/Turn_left_scenario_test.csv'
    df_turn_left_scenario = pd.read_csv(filepath_turn_left_scenario)
    df_all_seg_all_scenario_intersection_info = pd.DataFrame()
    scenario_all_count = 0
    all_intersection_info = []
    target_scenario = 18
    target_segment = 46
    length = 80   #目标交叉口的距离范围，需要进行调整
    for i in tqdm(range(len(file_index_list))):
        single_segment_all_scenario_all_lane_topo_info = []
        file_index = file_index_list[i]
        segment_file = all_file_list[i]
        print('Now is the file:%s' % file_index)
        #轨迹信息提取
        filepath_trj = 'data_save/all_scenario_all_objects_info/'+ file_index +'_all_scenario_all_object_info_1.csv'
        seg_trj = pd.read_csv(filepath_trj)
        single_seg_all_scenario_id = pd.unique(seg_trj['scenario_label'])


        #-------------Tf格式文件数据类型转换--------------------
        segment_dataset = tf.data.TFRecordDataset(segment_file)
        segment_dataset = segment_dataset.apply(tf.data.experimental.ignore_errors())

        #-------------静态点数据提取-----------
        df_single_seg_all_scenario_map_point_info = Ext.map_lane_point_extract_single_seg(segment_dataset,file_index,test_state,target_scenario)
        outpath_map_point = 'data/' + file_index + '_scenario_' + str(target_scenario) + '_static_map_info.csv'
        df_single_seg_all_scenario_map_point_info.to_csv(outpath_map_point)
        #-------------静态点数据提取结束-----------


        #------------地图拓扑信息提取--------------
        # df_single_seg_all_scenario_lane_topo_info = Ext.get_single_seg_all_scenario_lane_topo_info(segment_dataset,file_index,seg_trj,length,test_state,target_scenario)   #提取一个segment中所有scenario的地图场景中车道线的拓扑信息
        # outpath_lane_topo_info = 'data/all_lane_topo_info/' + file_index + '_all_lane_topo_info' + '.csv'  # 现在是一个scenario导出一个csv,后面需要导出一个完整的
        # df_single_seg_all_scenario_lane_topo_info.to_csv(outpath_lane_topo_info)
        #-----------地图拓扑信息提取结束-----------

        #-----------交叉口信息提取-----------------
        df_single_seg_all_intersection_info, df_single_seg_all_scenario_lane_topo_info = Ext.get_intersection_info(
            segment_dataset, df_turn_left_scenario,file_index, seg_trj, length, test_state, target_scenario=-1)
        df_all_seg_all_scenario_intersection_info = pd.concat(
            [df_all_seg_all_scenario_intersection_info, df_single_seg_all_intersection_info], axis=0)
        outpath_lane_topo_info = 'data/all_lane_topo_info/' + file_index + '_all_lane_topo_info' + '.csv'  # 现在是一个scenario导出一个csv,后面需要导出一个完整的
        df_single_seg_all_scenario_lane_topo_info.to_csv(outpath_lane_topo_info)
        #------------交叉口信息提取结束-------------


        #-----------动态信息提取与判断--------------
        scenario_label = 18
        dynamic_map_target_lane = Ext.get_dynamic_map_target_lane(file_index, scenario_label)




    outpath_intersection_info = 'data/' + 'all_seg_all_intersection_info' + '.csv'  # 现在是一个scenario导出一个csv,后面需要导出一个完整的
    df_all_seg_all_scenario_intersection_info.to_csv(outpath_intersection_info)





