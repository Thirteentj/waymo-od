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
import Extract01 as Ext




if __name__ == '__main__':
    #-----------------------load_data ---------------------

    test_state = 1
    filepath_oridata = 'E:/waymo_motion_dataset/training_20s.tfrecord-*-of-01000'
    all_file_list, file_index_list = Ext.get_file_list(filepath_oridata)
    scenario_all_count = 0
    all_intersection_info = []
    length = 80   #目标交叉口的距离范围，需要进行调整

    target_segment,target_scenario = 9,7

    for i in tqdm(range(len(file_index_list))):

        single_segment_all_scenario_all_lane_topo_info = []
        file_index = file_index_list[i]
        segment_file = all_file_list[i]
        print('Now is the file:%s' % file_index)
        if file_index == '00000':
            segment_id = 0
        else:
            segment_id = eval(file_index.strip('0'))

        if test_state == 1 :
            if segment_id < target_segment:
                continue
            elif segment_id > target_segment:
                break

        filepath_trj = 'E:/Result_save/data_save/all_scenario_all_objects_info/'+ file_index +'_all_scenario_all_object_info_1.csv'
        seg_trj = pd.read_csv(filepath_trj)
        single_seg_all_scenario_id = pd.unique(seg_trj['scenario_label'])
        segment_name_list = []
        segment_dataset = tf.data.TFRecordDataset(segment_file)
        segment_dataset = segment_dataset.apply(tf.data.experimental.ignore_errors())

        scenario_label = 0  # 所有场景的ID数量记录
        df_single_seg_all_scenario_lane_topo_info = pd.DataFrame()
        df_single_seg_all_scenario_map_point_info = pd.DataFrame()  # 静态地图所有车道线的散点信息提取

        #single_file_all_scenario = []
        for one_scenario in segment_dataset:  # one_scenario 就是一个scenario
            single_scenario_all_feature = []
            scenario_label += 1
            scenario_all_count += 1
            #print('Now is the scenario:%s' % scenario_label)
            if test_state == 1:
                #intersection_center_loc = (1798.4981814285616, -2136.2457659217735)
                if  scenario_label < target_scenario:
                    continue
                elif scenario_label > target_scenario:
                    break
            scenario = Scenario()
            scenario.ParseFromString(one_scenario.numpy())  # 数据格式转化
            time_stamp_all = scenario.timestamps_seconds
            map_features = scenario.map_features


            map_features_id_list = []
            segment_id = Ext.file_index_to_segment_id(file_index)

            scenario_trj = seg_trj[seg_trj['scenario_label'] == scenario_label]
            df_single_scenario_map_point = Ext.map_lane_point_extract_single_scenaraio(map_features, file_index,scenario_label)  # 提取地图中所有车道的散点信息
            df_single_seg_all_scenario_map_point_info = pd.concat(
                [df_single_seg_all_scenario_map_point_info, df_single_scenario_map_point], axis=0)  # 纵向方向合并df

            single_scenario_all_lane_entry_exit_info, single_map_dict, lane_turn_left_id, lane_turn_right_id, single_scenario_all_feature, map_features_id_list = Ext.map_topo_info_extract(
                map_features, single_scenario_all_feature, map_features_id_list, file_index, scenario_label)  # 车道拓扑关系提取

            df_single_scenario_lane_topo_info = pd.DataFrame(single_scenario_all_lane_entry_exit_info)
            df_single_seg_all_scenario_lane_topo_info = pd.concat(
                [df_single_seg_all_scenario_lane_topo_info, df_single_scenario_lane_topo_info], axis=0)

            #dynamic_map_target_lane = Ext.get_dynamic_map_target_lane(file_index,scenario_label)
            fig_save_path = 'E:/Result_save/figure_save/topo_fig_correct/'
            road_edge_count, lane_count, road_line, all_element_count = Ext.plot_top_view_single_pic_map_2(scenario_trj, file_index, scenario_label,
                                                                                                           scenario, lane_turn_left_id, lane_turn_right_id,fig_save_path)

        # #-----------------ouput------------------
        # outpath_lane_topo_info = 'E:/Result_save/data_save/all_lane_topo_info/' + file_index +'_all_lane_topo_info_all_intersections' + '.csv'  #现在是一个scenario导出一个csv,后面需要导出一个完整的
        # #print(outpath_lane_topo_info)
        # df_single_seg_all_scenario_lane_topo_info.to_csv(outpath_lane_topo_info)
        # outpath_map_point = 'E:/Result_save/data_save/static_map_point_info/' + file_index + '_all_scenario_static_map_info.csv'
        # df_single_seg_all_scenario_map_point_info.to_csv(outpath_map_point)
