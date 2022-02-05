import os
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import matplotlib as mpl
import cv2
import glob
from scipy.interpolate import CubicSpline
import scipy.interpolate
from scipy import signal
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from gekko import GEKKO
import pywt
# import waymo dataset related modules
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from tqdm import tqdm
from waymo_open_dataset.protos.scenario_pb2 import Scenario
import time


def veh_trj_collect(scenario,segment_id,scenario_label):
    single_segment_all_scenario = []
    time_stamp_all = scenario.timestamps_seconds  #整个时间片段的列表
    single_scenario_AV_index = scenario.sdc_track_index   #单个场景下的自动驾驶车辆的ID
    tracks_all = scenario.tracks
    tracks_label = 0
    object_of_interest = scenario.objects_of_interest
    #print(scenario_label)
    for single_track in tracks_all:
        state_index = 0
        tracks_label += 1
        for single_state in single_track.states:
            single_scenario_single_track = {}  # 单个场景下一辆车的所有轨迹信息，包含整个9s
            single_scenario_single_track['segment_id'] = segment_id
            single_scenario_single_track['scenario_label'] = scenario_label
            single_scenario_single_track['tracks_label'] = tracks_label
            single_scenario_single_track['obj_id'] = single_track.id
            single_scenario_single_track['obj_type'] = single_track.object_type
            #TYPE_UNSET = 0;
            #TYPE_VEHICLE = 1;
            #TYPE_PEDESTRIAN = 2;
            #TYPE_CYCLIST = 3;
            #TYPE_OTHER = 4;
            if single_track.id == single_scenario_AV_index:  #判断是否为AV
                single_scenario_single_track['is_AV'] = 1
            else:
                single_scenario_single_track['is_AV'] = 0
            if single_track.id in object_of_interest:
                single_scenario_single_track['is_interest'] = 1
            else:
                single_scenario_single_track['is_interest'] = 0
            single_scenario_single_track['time_stamp'] = time_stamp_all[state_index]
            #single_scenario_single_track['dynamic_map_states']  = scenario.dynamic_map_states[state_index]  #动态地图信息状态，大多数为空
            single_scenario_single_track['frame_label'] = state_index + 1
            single_scenario_single_track['valid'] = single_state.valid
            if single_state.valid == True:
                single_scenario_single_track['center_x'] = single_state.center_x
                single_scenario_single_track['center_y'] = single_state.center_y
                single_scenario_single_track['center_z'] = single_state.center_z
                single_scenario_single_track['length'] = single_state.length
                single_scenario_single_track['width'] = single_state.width
                single_scenario_single_track['height'] = single_state.height
                single_scenario_single_track['heading'] = single_state.heading
                single_scenario_single_track['velocity_x'] = single_state.velocity_x
                single_scenario_single_track['velocity_y'] = single_state.velocity_y
            state_index += 1
            single_segment_all_scenario.append(single_scenario_single_track)

    return single_segment_all_scenario

#Generate visualization images.
def create_figure_and_axes(size_pixels):
    """Initializes a unique figure and axes for plotting."""
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

    # Sets output image to pixel resolution.
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    fig.set_tight_layout(True)
    ax.grid(False)
    return fig, ax


def plot_top_view_single_pic(trj_in, scenario_id_in, frame_id_in):
    # this function plots one single scenario of the top view video
    trj_in['center_x'] = trj_in['center_x'] - trj_in['center_x'].min()
    trj_in['center_y'] = trj_in['center_y'] - trj_in['center_y'].min()
    unique_veh_id = pd.unique(trj_in['obj_id'])
    plt.figure(figsize=(18, 13.5))
    plt.figure()
    plt.xlabel('global center x (m)', fontsize=10)
    plt.ylabel('global center y (m)', fontsize=10)
    plt.axis('square')
    plt.xlim([trj_in['center_x'].min() - 1, trj_in['center_x'].max() + 1])
    plt.ylim([trj_in['center_y'].min() - 1, trj_in['center_y'].max() + 1])
    title_name = 'Scenario ' + str(scenario_id_in)
    plt.title(title_name, loc='left')
    plt.xticks(
        np.arange(round(float(trj_in['center_x'].min())), round(float(trj_in['center_x'].max())), 10),
        fontsize=5)
    plt.yticks(
        np.arange(round(float(trj_in['center_y'].min())), round(float(trj_in['center_y'].max())), 10),
        fontsize=5)
    ax = plt.gca()

    #print(trj_in)
    #print(trj_in[trj_in['is_AV'] == 1])
    av_veh_trj = trj_in[trj_in['is_AV'] == 1]

    #av_current_heading = av_veh_trj.loc[av_veh_trj['frame_label'] == frame_id_in,'heading'].values[0]

    for single_veh_id in unique_veh_id:
        single_veh_trj = trj_in[trj_in['obj_id'] == single_veh_id]
        single_veh_trj = single_veh_trj[single_veh_trj['frame_label'] == frame_id_in]
        #print(single_veh_trj)
        if len(single_veh_trj) > 0 and single_veh_trj['valid'].iloc[0] == True:
            ts = ax.transData
            coords = [single_veh_trj['center_x'].iloc[0],single_veh_trj['center_y'].iloc[0]]
            if single_veh_trj['is_AV'].iloc[0] == 1:
                temp_facecolor = 'black'
                temp_alpha = 0.99
                heading_angle = single_veh_trj['heading'].iloc[0] * 180 / np.pi
                tr = mpl.transforms.Affine2D().rotate_deg_around(coords[0], coords[1], heading_angle)
            else:
                if single_veh_trj['is_interest'].iloc[0] == 1:
                    temp_facecolor = 'red'   #有交互行为的车辆变为黑色
                else:
                    if single_veh_trj['obj_type'].iloc[0] == 1:
                        temp_facecolor = 'blue'
                    elif single_veh_trj['obj_type'].iloc[0] == 2:
                        temp_facecolor = 'green'
                    else:
                        temp_facecolor = 'magenta'
                temp_alpha = 0.5
                heading_angle = single_veh_trj['heading'].iloc[0] * 180 / np.pi
                # transform for other vehicles, note that the ego global heading should be added to current local heading
                tr = mpl.transforms.Affine2D().rotate_deg_around(coords[0], coords[1], heading_angle)
            t = tr + ts
            # note that exact xy needs to to calculated
            veh_length = single_veh_trj['length'].iloc[0]
            veh_width = single_veh_trj['width'].iloc[0]
            ax.add_patch(patches.Rectangle(
                xy=(single_veh_trj['center_x'].iloc[0] - 0.5 * veh_length,
                    single_veh_trj['center_y'].iloc[0] - 0.5 * veh_width),
                width=veh_length,
                height=veh_width,
                linewidth=0.1,
                facecolor=temp_facecolor,
                edgecolor='black',
                alpha=temp_alpha,
                transform=t))
            # add vehicle local id for only vehicle object
            if single_veh_trj['obj_type'].iloc[0] == 1:
                temp_text = plt.text(single_veh_trj['center_x'].iloc[0],
                                     single_veh_trj['center_y'].iloc[0], str(single_veh_id), style='italic',
                                     weight='heavy', ha='center', va='center', color='white', rotation=heading_angle,
                                     size=3)
                temp_text.set_path_effects(
                    [path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
    trj_save_name = 'figure_save/temp_top_view_figure/top_view_segment_' + '__'+'scenario_'+str(scenario_id_in) + '_frame_' + str(
        frame_id_in) + '_trajectory.jpg'
    plt.savefig(trj_save_name, dpi=600)
    plt.close('all')


def top_view_video_generation(path_2,single_seg_id_in,scenario_id_in):
    # this function generates one top view video based on top view figures from one segment
    img_array = []
    for num in range(1, len(os.listdir('figure_save/temp_top_view_figure/')) + 1):
        image_filename = 'figure_save/temp_top_view_figure/' + 'top_view_segment_' +'__'+'scenario_'+str(
            scenario_id_in) + '_frame_' + str(num) + '_trajectory.jpg'
        img = cv2.imread(image_filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    video_save_name = 'figure_save/top_view_video/' + path_2+'_animation_top_view_segment_' + str(single_seg_id_in) +'_scenario_'+str(scenario_id_in)+ '.avi'
    out = cv2.VideoWriter(video_save_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('No %d top view video made success'%scenario_id_in)
    # after making the video, delete all the frame jpgs
    filelist = glob.glob(os.path.join('figure_save/temp_top_view_figure/', "*.jpg"))
    for f in filelist:
        os.remove(f)



#   *********************** Stage 1 : collect tracks information from original dataset
#path_2_list = ['00043','00044','00045','00046','00047','00048','00049']
path_2_list = ['00044']
segment_id = 0
for path_2 in path_2_list:
    #目前仅对一个文件进行测试

    print('Now is the file:%s'%path_2)
    segment_name_list = []

    #uncompressed_scenario_training_20s_training_20s.tfrecord-00048-of-01000

    interactive_behavior = pd.DataFrame(columns=['id','time_segment','veh_1','veh_2'])   #记录所有的交互行为，记录交互的时间范围以及两辆车的id信息
    #path_1 = 'D:/Data/WaymoData/uncompressed_scenario_training_training.tfrecord-'
    path_1 = 'D:/Data/WaymoData/uncompressed_scenario_training_20s_training_20s.tfrecord-'
    path_3 = '-of-01000'
    segment_file = path_1+path_2+path_3
    #segment_file = 'D:/Data/WaymoData/uncompressed_tf_example_testing_interactive_testing_interactive_tfexample.tfrecord-00000-of-00150'
    #segment_file = 'D:/Data/WaymoData/uncompressed_scenario_training_training.tfrecord-00000-of-01000'
    segment_id += 1

    segment_name = os.path.basename(segment_file)
    single_segment_name_dict = {}  #对1个tfrecord数据集进行记录
    single_segment_name_dict['segment_id'] = segment_id
    single_segment_name_dict['segment_file_name'] = segment_name
    filename = segment_file

    segment_dataset = tf.data.TFRecordDataset(filename)
    segment_dataset = segment_dataset.apply(tf.data.experimental.ignore_errors())

    scenario_label = 0  #所有场景的ID数量记录
    scenario_valid_label = 0   #包含交互行为的场景ID行为记录
    single_segment_scenario_valid_dict = {}  #对于单个片段，所有有效的场景的车辆ID信息记录
    all_segment_scenario_valid_dict = []  #所有车辆

    all_segment_all_scenario_all_object_info = []
    for one_record in segment_dataset:  #one_record 就是一个scenario
        single_segment_all_scenario = []  # 单个片段的所有场景信息
        scenario_label += 1
        scenario = Scenario()
        scenario.ParseFromString(one_record.numpy())  #数据格式转化
        #print(scenario.timestamps_seconds)

        #data = open("D:/Data/WaymoData/test.txt", 'w+')
        #print(scenario, file=data)
        #data.close()
        #break
    




        if scenario.objects_of_interest != []:  #对包含交互行为的scenario 进行记录
            scenario_valid_label += 1
            single_segment_scenario_valid_dict['segment_id'] = segment_id
            single_segment_scenario_valid_dict['file_id'] = path_2
            single_segment_scenario_valid_dict['scenario_valid_label'] = scenario_valid_label
            single_segment_scenario_valid_dict['scenario_label'] = scenario_label
            single_segment_scenario_valid_dict['scenario_id'] = scenario.scenario_id
            single_segment_scenario_valid_dict['objects_of_interest'] = scenario.objects_of_interest  #一个列表，包含两个交互对象的ID
            single_valid_scenario_time_range = str(scenario.timestamps_seconds[0]) + '-' + str(scenario.timestamps_seconds[-1])
            single_segment_scenario_valid_dict['time_range'] = single_valid_scenario_time_range
            all_segment_scenario_valid_dict.append(single_segment_scenario_valid_dict)
            single_segment_scenario_valid_dict = {}
            #print(scenario.objects_of_interest)
            #print(scenario.timestamps_seconds)
        print(scenario_label)
        # -----------------trajectory extraction -----------------
        single_segment_all_scenario = veh_trj_collect(scenario,segment_id,scenario_label)  #返回单个scenario的所有场景信息
        all_segment_all_scenario_all_object_info += single_segment_all_scenario


        save_step = 50
        if scenario_label % save_step == 0:
            print('print')
            filename = 'data_save/'+ path_2 + '_all_object_all_scenario_test' + '_' + str((scenario_label // save_step)) + '.csv'
            all_segment_all_scenario_all_object_info_pd = pd.DataFrame(all_segment_all_scenario_all_object_info)
            all_segment_all_scenario_all_object_info_pd.to_csv(filename, index=False)
            all_segment_all_scenario_all_object_info = []

        #if scenario_label ==200:
        #    break
        # print(scenario_label)
    # print(scenario_label,scenario_valid_label)  #第一个文件中有517个场景，其中215个有交互行为
    filename = 'data_save/'+ path_2 + '_all_object_all_scenario_test' + '_' + str((scenario_label // save_step)+1) + '.csv'
    all_segment_all_scenario_all_object_info_pd = pd.DataFrame(all_segment_all_scenario_all_object_info)
    all_segment_all_scenario_all_object_info_pd.to_csv(filename, index=False)
    all_segment_all_scenario_all_object_info = []


    filename_2 = 'data_save/'+path_2+ 'all_object_all_scenario_objects_of_interest_valid_dict_test'  + '.csv'
    all_segment_scenario_valid_dict_pd = pd.DataFrame(all_segment_scenario_valid_dict)
    all_segment_scenario_valid_dict_pd.to_csv(filename_2, index=False)

    # ********************Stage 2: visulization of information  **********************

    # ---------- process calulated trajectories from the csv lidar information file ----------
    #trj_read_name = 'data_save/all_object_all_scenario_test_1.csv'
    trj_read_name = 'data_save/'+ path_2 + '_all_object_all_scenario_test' + '_' + '1' + '.csv'
    print(trj_read_name)
    temp_trj = pd.read_csv(trj_read_name,header=0,dtype={'global_center_x': np.float64, 'global_center_y': np.float64})
    all_segment_id = pd.unique(temp_trj['segment_id'])   #一个segment中有数百个scenario，一个scenario中又有200帧frame
    filelist = glob.glob(os.path.join('figure_save/temp_top_view_figure/','*.jpg'))
    for f in filelist:
        os.remove(f)
    for single_seg_id in all_segment_id:   #这里的一个scenario相当于之前的一个segment
        seg_trj = temp_trj[temp_trj['segment_id'] == single_seg_id]
        single_seg_all_scenario_id = pd.unique(seg_trj['scenario_label'])
        for i in tqdm(range(len(single_seg_all_scenario_id))):  #一个scenario 生成一个video
            #start_time = time.time()
            single_scenario_id = single_seg_all_scenario_id[i]
            scenario_trj = seg_trj[seg_trj['scenario_label'] == single_scenario_id]
            #目前只选出包含AV车辆的scenario
            #is_av = pd.unique(scenario_trj['is_AV'])
            #if 1 in is_av:   #该scenario 中有AV
            scenario_print = 'Top view video now in segment:' + str(single_seg_id) + '  in scenario: ' + str(single_scenario_id)
            print(scenario_print)

            #top_view_trj = scenario_trj[['obj_id', 'obj_type', 'scenario_label', 'time_stamp', 'center_x', 'center_y','length', 'width', 'heading','frame_label','is_AV','valid']]
            top_view_trj = scenario_trj
            total_frame_num = scenario_trj['frame_label'].max()
            for frame_id in range(1,total_frame_num+1):
                if frame_id == 1:
                    current_frame_trj = top_view_trj[top_view_trj['frame_label'] == frame_id]
                plot_top_view_single_pic(top_view_trj,single_scenario_id,frame_id)
            print('No.%d fig has been made,now begin to generate top view viedo.'%single_scenario_id)
            #----------video generation------------
            top_view_video_generation(path_2,single_seg_id,single_scenario_id)
            #end_time = time.time()
            #print('This scenario takes time:%f'%(end_time-start_time))


