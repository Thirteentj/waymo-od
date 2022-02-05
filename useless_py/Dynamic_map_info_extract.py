import os
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import glob
import seaborn as sns
# import waymo dataset related modules
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from tqdm import tqdm
from waymo_open_dataset.protos.scenario_pb2 import Scenario

def get_file_list(filepath):

    all_files = sorted(glob.glob(filepath))
    segs_name_index = []
    for file in all_files:
        segment_name = os.path.basename(file)
        segs_name_index.append(segment_name[-14:-9])
    #print(segs_name_all)
    print(segs_name_index)
    return all_files,segs_name_index

def get_trafficlights_state(state_type):
    #print(state_type)
    state_type = int(state_type)
    state_dic = {1:'LANE_STATE_ARROW_STOP',2:'LANE_STATE_ARROW_CAUTION',3:'LANE_STATE_ARROW_GO',
                 4:'LANE_STATE_STOP',5:'LANE_STATE_CAUTION',6:'LANE_STATE_GO',
                 7:'LANE_STATE_FLASHING_STOP',8:'LANE_STATE_FLASHING_CAUTION'}

    if state_type == 1:
        state = 'LANE_STATE_ARROW_STOP'
    elif state_type == 2:
        state = 'LANE_STATE_ARROW_CAUTION'
    elif state_type == 3:
        state = 'LANE_STATE_ARROW_GO'
    elif state_type == 4:
        state = 'LANE_STATE_STOP'
    elif state_type == 5:
        state = 'LANE_STATE_CAUTION'
    elif state_type == 6:
        state = 'LANE_STATE_GO'
    elif state_type == 7:
        state = 'LANE_STATE_FLASHING_STOP'
    elif state_type == 8:
        state = 'LANE_STATE_FLASHING_CAUTION'
    else:
        state = 'LANE_STATE_UNKNOWN'

    #print(type(state_type))
    #print(state_type==4)
    #state = state_dic[state_type]
    return state


if __name__ == "__main__":


    test_state = 0
    filepath = 'D:/LJQ/WaymoData_motion_1/training_20s.tfrecord-*-of-01000'
    all_file_list, file_index_list = get_file_list(filepath)
    empty_dynamic_map_count = 0
    scenario_all_count = 0
    for i in tqdm(range(len(file_index_list))):
        file_index = file_index_list[i]
        segment_file = all_file_list[i]
        #print('Now is the file:%s' % file_index)
        segment_name_list = []
        dynamic_map_single_file_all_scenario = []   #交通信号灯数据
        segment_dataset = tf.data.TFRecordDataset(segment_file)
        segment_dataset = segment_dataset.apply(tf.data.experimental.ignore_errors())

        scenario_label = 0  # 所有场景的ID数量记录
        if i <= 478:
            continue
        for one_scenario in segment_dataset:  # one_scenario 就是一个scenario

            scenario_label += 1
            scenario_all_count += 1
            #print("Scenario %d" % scenario_label)
            if test_state == 1:
                if scenario_label == 3:
                    break
            scenario = Scenario()
            scenario.ParseFromString(one_scenario.numpy())  # 数据格式转化
            time_stamp_all = scenario.timestamps_seconds
            map_features = scenario.map_features
            length = len(time_stamp_all)
            map_features_id_list = []
            dynamic_map_states_id_list = []
            dynamic_map_states = scenario.dynamic_map_states     #每个时间戳都对应一个dynamic_map_states 一个scenario共199个，一个时间戳里的states包含所有车道的信号灯信息
            one_state = list(dynamic_map_states[0].lane_states)   #取出来一个看是否为空，如果第一个为空那么这个场景应该全部为空
            for single_feature in map_features:
                map_features_id_list.append(single_feature.id)
            #print(map_features_id_list)
            dynamic_index = 0
            dynamic_index_single_scenario_single_timestamp = {}
            if one_state != []:
                for single_dynamic_map_states in dynamic_map_states:
                    #print(single_dynamic_map_states)
                    for lane_states in single_dynamic_map_states.lane_states:
                        #print(lane_states)
                        dynamic_index_single_scenario_single_timestamp['file_index'] = file_index
                        dynamic_index_single_scenario_single_timestamp['scenario_index'] = scenario_label
                        try:
                            dynamic_index_single_scenario_single_timestamp['time_stamp'] = time_stamp_all[dynamic_index]
                        except:
                            continue
                        dynamic_index_single_scenario_single_timestamp['lane'] = lane_states.lane
                        dynamic_index_single_scenario_single_timestamp['state_type'] = lane_states.state
                        dynamic_index_single_scenario_single_timestamp['state'] = get_trafficlights_state(lane_states.state)
                        dynamic_index_single_scenario_single_timestamp['stop_point_x'] = lane_states.stop_point.x
                        dynamic_index_single_scenario_single_timestamp['stop_point_y'] = lane_states.stop_point.y
                        dynamic_index_single_scenario_single_timestamp['stop_point_z'] = lane_states.stop_point.z
                        dynamic_map_single_file_all_scenario.append(dynamic_index_single_scenario_single_timestamp)
                        dynamic_index_single_scenario_single_timestamp = {}
                    dynamic_index += 1
                    #print(dynamic_index)
            else:    #表明该场景没有信号灯数据
                empty_dynamic_map_count += 1
                #print("No.%d scenario doesn't have dynamic map features" % empty_dynamic_map_count)
        #print(dynamic_map_single_file_all_scenario)
        df_out = pd.DataFrame(dynamic_map_single_file_all_scenario)
        outpath = '../Result_save/data_save/dynamic_map_info/'+ file_index + '_dynamic_map_single_file_all_scenario' +'.csv'
        df_out.to_csv(outpath)
    print("There are {} scenarios at all, and {} scenarios don't have dynamic map infomation".format(scenario_all_count,empty_dynamic_map_count))

    '''
    print(len(time_stamp_all))
    #print(len(dynamic_map_states))
    #print(dynamic_map_states)
    #print(dynamic_map_states[0],dynamic_map_states[1],dynamic_map_states[2])
    #print(dynamic_map_states[0].lane_states)
    #print(type(dynamic_map_states[0]))
    
    if one_state != []:
        print(type(one_state[0]))
        print(one_state[0])
    
    
    
    if dynamic_map_states == []:
        empty_dynamic_map_count
        print("No.%d scenario doesn't have dynamic map features"%empty_dynamic_map_count)
    print("Scenario %d"%scenario_label)
    
    # print(scenario.timestamps_seconds)
    output = "D:/Data/WaymoData/"+"scenario_"+str(scenario_label)+".txt"
    data = open(output, 'w+')
    print(scenario, file=data)
    data.close()
    # break
    '''
