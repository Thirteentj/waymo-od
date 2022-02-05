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
import tqdm

def veh_trj_collect(frame_in):
    global all_segment_all_frame_all_object_info
    # this funtion collects all Lidar object information in current frame
    # collect environment context in this frame
    frame_context_dict = frame_context_update(frame)
    # print(frame_context_dict)
    ego_row_dict = frame_context_dict.copy()  # add context info to every row
    # collect ego (AV) vehicle's timestamp, position and speed
    ego_row_dict['obj_type'] = 'vehicle'
    ego_row_dict['obj_id'] = 'ego'
    ego_row_dict['global_time_stamp'] = frame_in.timestamp_micros  # unix time (in micro seconds)
    # time referenced to segment start time
    ego_row_dict['local_time_stamp'] = (frame_in.timestamp_micros - segment_start_time) / float(1000000)  # in seconds
    # self driving car's (sdc) global position and heading (yaw, pitch, roll)
    sdc_pose = frame_in.pose.transform  # the transformation matrix
    # print(sdc_pose)
    frame_images = frame_in.images
    for image in frame_images:
        # print(image.velocity)
        ego_speed_x = image.velocity.v_x
        ego_speed_y = image.velocity.v_y
        ego_angular_speed = image.velocity.w_z
        # only get speed from the front camera
        break
    # print(image.pose)
    # ego_velocity = frame_in.images
    # ego vehicle's local position will be 0, because itself is the origin
    ego_row_dict['local_center_x'] = 0
    ego_row_dict['local_center_y'] = 0
    ego_row_dict['local_center_z'] = 0
    # ego vehicle's global position is extracted from the transformation matrix
    ego_row_dict['global_center_x'] = sdc_pose[3]
    ego_row_dict['global_center_y'] = sdc_pose[7]
    ego_row_dict['global_center_z'] = sdc_pose[11]
    # note that the actual model of AV is 2019 Chrysler Pacifica Hybrid
    # the dimensions for AV is length 5.18m, width 2.03m, height 1.78m
    ego_row_dict['length'] = 5.18
    ego_row_dict['width'] = 2.03
    ego_row_dict['height'] = 1.78
    ego_row_dict['heading'] = math.atan2(sdc_pose[4], sdc_pose[0])
    ego_row_dict['speed_x'] = ego_speed_x
    ego_row_dict['speed_y'] = ego_speed_y
    # accleration remains to be calculated
    ego_row_dict['accel_x'] = 0
    ego_row_dict['accel_y'] = 0
    ego_row_dict['angular_speed'] = ego_angular_speed
    # print(ego_row_dict)
    # add to final file
    all_segment_all_frame_all_object_info.append(ego_row_dict)
    # collect vehicle's info in the lidar label

    for lidar_label in frame_in.laser_labels:   #lidar标签中包含的是所有周围运动对象的信息，之前完成的自动驾驶主车的信息提取
        # labe object type and its correspoding
        # TYPE_UNKNOWN = 0;
        # TYPE_VEHICLE = 1;
        # TYPE_PEDESTRIAN = 2;
        # TYPE_SIGN = 3;
        # TYPE_CYCLIST = 4;
        if lidar_label.type in [1, 2, 4]:
            temp_row_dict = ego_row_dict.copy()
            if lidar_label.type == 1:
                temp_row_dict['obj_type'] = 'vehicle'
            elif lidar_label.type == 4:
                temp_row_dict['obj_type'] = 'bicycle'
            else:
                temp_row_dict['obj_type'] = 'pedestrian'
            temp_row_dict = collect_lidar_veh_label(lidar_label, temp_row_dict, ego_row_dict, sdc_pose)
            # add to final file
            all_segment_all_frame_all_object_info.append(temp_row_dict)


def collect_lidar_veh_label(single_lidar_label, row_dict, ego_dict, ego_pose):
    # this function extract the information of a single object (Lidar label)
    # note that the original position and heading in label is in local coordinate
    # single_lidar_label is from lidar label from original data
    # row_dict is an initialized dictionary that will be filled
    # global unique object_id
    row_dict['obj_id'] = single_lidar_label.id
    row_dict['local_center_x'] = single_lidar_label.box.center_x
    row_dict['local_center_y'] = single_lidar_label.box.center_y
    row_dict['local_center_z'] = single_lidar_label.box.center_z
    # we need to use ego_dict and ego_pose to transform local label position to global position
    # (in vehicle frame), it needs to be transformed to global frame
    # make ego_pose in the form of transformation matrix
    trans_matrix = np.reshape(np.array(ego_pose), (4, 4))
    # print(trans_matrix)
    local_pos_matrix = np.reshape(
        np.array([row_dict['local_center_x'], row_dict['local_center_y'], row_dict['local_center_z'], 1]), (4, 1))  #需要运行调试
    # print(local_pos_matrix)
    label_global_pos = np.matmul(trans_matrix, local_pos_matrix)  #矩阵乘法
    # print(label_global_pos)
    row_dict['global_center_x'] = label_global_pos[0][0]
    row_dict['global_center_y'] = label_global_pos[1][0]
    row_dict['global_center_z'] = label_global_pos[2][0]
    row_dict['length'] = single_lidar_label.box.length
    row_dict['width'] = single_lidar_label.box.width
    row_dict['height'] = single_lidar_label.box.height
    frame_ego_heading = ego_dict['heading']
    row_dict['heading'] = single_lidar_label.box.heading + frame_ego_heading
    row_dict['speed_x'] = single_lidar_label.metadata.speed_x
    row_dict['speed_y'] = single_lidar_label.metadata.speed_y
    row_dict['accel_x'] = single_lidar_label.metadata.accel_x
    row_dict['accel_y'] = single_lidar_label.metadata.accel_y
    # angular speed remains to be calculated
    row_dict['angular_speed'] = 0
    return row_dict



def frame_context_update(frame_in):
    # collect environment context in this frame
    frame_context_dict = {}
    frame_context_dict['segment_id'] = segment_id
    frame_context_dict['frame_label'] = frame_label
    #print(frame_in.context)
    frame_context_dict['time_of_day'] = frame_in.context.stats.time_of_day
    frame_context_dict['location'] = frame_in.context.stats.location
    frame_context_dict['weather'] = frame_in.context.stats.weather
    for count in frame_in.context.stats.laser_object_counts:
        if count.type != 1:  # note that 1 means vehicle object
            continue
        frame_context_dict['laser_veh_count'] = count.count
    return frame_context_dict



def final_trj_result_format():
    # format the final output
    global all_segment_all_frame_all_object_info_pd
    all_segment_all_frame_all_object_info_pd['local_time_stamp'] = all_segment_all_frame_all_object_info_pd[
        'local_time_stamp'].map('{:.2f}'.format)
    all_segment_all_frame_all_object_info_pd['local_center_x'] = all_segment_all_frame_all_object_info_pd[
        'local_center_x'].map('{:.4f}'.format)
    all_segment_all_frame_all_object_info_pd['local_center_y'] = all_segment_all_frame_all_object_info_pd[
        'local_center_y'].map('{:.4f}'.format)
    all_segment_all_frame_all_object_info_pd['local_center_z'] = all_segment_all_frame_all_object_info_pd[
        'local_center_z'].map('{:.4f}'.format)
    all_segment_all_frame_all_object_info_pd['global_center_x'] = all_segment_all_frame_all_object_info_pd[
        'global_center_x'].map('{:.4f}'.format)
    all_segment_all_frame_all_object_info_pd['global_center_y'] = all_segment_all_frame_all_object_info_pd[
        'global_center_y'].map('{:.4f}'.format)
    all_segment_all_frame_all_object_info_pd['global_center_z'] = all_segment_all_frame_all_object_info_pd[
        'global_center_z'].map('{:.4f}'.format)
    all_segment_all_frame_all_object_info_pd['length'] = all_segment_all_frame_all_object_info_pd['length'].map(
        '{:.2f}'.format)
    all_segment_all_frame_all_object_info_pd['width'] = all_segment_all_frame_all_object_info_pd['width'].map(
        '{:.2f}'.format)
    all_segment_all_frame_all_object_info_pd['height'] = all_segment_all_frame_all_object_info_pd['height'].map(
        '{:.2f}'.format)
    all_segment_all_frame_all_object_info_pd['heading'] = all_segment_all_frame_all_object_info_pd['heading'].map(
        '{:.4f}'.format)
    all_segment_all_frame_all_object_info_pd['speed_x'] = all_segment_all_frame_all_object_info_pd['speed_x'].map(
        '{:.4f}'.format)
    all_segment_all_frame_all_object_info_pd['speed_y'] = all_segment_all_frame_all_object_info_pd['speed_y'].map(
        '{:.4f}'.format)
    all_segment_all_frame_all_object_info_pd['accel_x'] = all_segment_all_frame_all_object_info_pd['accel_x'].map(
        '{:.4f}'.format)
    all_segment_all_frame_all_object_info_pd['accel_y'] = all_segment_all_frame_all_object_info_pd['accel_y'].map(
        '{:.4f}'.format)
    all_segment_all_frame_all_object_info_pd['angular_speed'] = all_segment_all_frame_all_object_info_pd[
        'angular_speed'].map('{:.4f}'.format)

def camera_video_generation():
    img_array = []
    #print(len(os.listdir('figure_save/temp_cam_pic/')))
    for num in range(1, len(os.listdir('figure_save/temp_cam_pic/')) + 1):
        image_filename = 'figure_save/temp_cam_pic/' + 'frame_' + str(num) + '.jpg'
        img = cv2.imread(image_filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    video_save_name = 'figure_save/cam_video/' + 'camera_video_segment_' + str(segment_id) + '.avi'
    out = cv2.VideoWriter(video_save_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('camera video made success')
    # after making the video, delete all the frame jpgs
    filelist = glob.glob(os.path.join('figure_save/temp_cam_pic/', "*.jpg"))
    for f in filelist:
        os.remove(f)

def show_camera_image(camera_image, camera_labels, layout, cmap=None):
    # Show a camera image and the given camera labels (if avaliable)
    ax = plt.subplot(*layout)
    # Draw the camera labels.
    for camera_labels in frame.camera_labels:
        # Ignore camera labels that do not correspond to this camera.
        if camera_labels.name != camera_image.name:
            continue
        # Iterate over the individual labels.
        for label in camera_labels.labels:
            # Draw the object bounding box.
            ax.add_patch(patches.Rectangle(
                xy=(label.box.center_x - 0.5 * label.box.length,
                    label.box.center_y - 0.5 * label.box.width),
                width=label.box.length,
                height=label.box.width,
                linewidth=1,
                edgecolor='red',
                facecolor='none'))
    # Show the camera image.
    frame_image = plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap, animated=True)
    plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
    plt.grid(False)
    plt.axis('off')
    return frame_image


def plot_top_view_ani_with_lidar_label(trj_in, seg_id_in, frame_id_in):
    # this function plots one single frame of the top view video
    # trj_in is a pandas with three columns(obj_id, frame_label, local_time_stamp, global_center_x, global_center_y, length, width, heading)
    # trj_in is all the trajectories within one segment
    # seg_id_in is the current segment id
    trj_in['global_center_x'] = trj_in['global_center_x'] - trj_in['global_center_x'].min()
    trj_in['global_center_y'] = trj_in['global_center_y'] - trj_in['global_center_y'].min()
    unique_veh_id = pd.unique(trj_in['obj_id'])
    plt.figure(figsize=(18, 13.5))
    plt.figure()
    plt.xlabel('global center x (m)', fontsize=10)
    plt.ylabel('global center y (m)', fontsize=10)
    plt.axis('square')
    plt.xlim([trj_in['global_center_x'].min() - 1, trj_in['global_center_x'].max() + 1])
    plt.ylim([trj_in['global_center_y'].min() - 1, trj_in['global_center_y'].max() + 1])
    # max_range = max(trj_in['global_center_x'].max(), )
    title_name = 'Segment ' + str(seg_id_in)
    plt.title(title_name, loc='left')
    plt.xticks(
        np.arange(round(float(trj_in['global_center_x'].min())), round(float(trj_in['global_center_x'].max())), 10),
        fontsize=5)
    plt.yticks(
        np.arange(round(float(trj_in['global_center_y'].min())), round(float(trj_in['global_center_y'].max())), 10),
        fontsize=5)
    ax = plt.gca()
    # find out the global heading of ego vehicle first, use it to transform other vehicles' local heading to global heading
    ego_veh_trj = trj_in.loc[trj_in['obj_id'] == 'ego', :]
    ego_current_heading = ego_veh_trj.loc[ego_veh_trj['frame_label'] == frame_id, 'heading'].values[0]
    # get all the trajectories until current frame
    for signle_veh_id in unique_veh_id:
        single_veh_trj = trj_in[trj_in['obj_id'] == signle_veh_id]
        # print(single_veh_trj)
        single_veh_trj = single_veh_trj[single_veh_trj['frame_label'] == frame_id_in]
        # print(single_veh_trj)
        if len(single_veh_trj) > 0:
            ts = ax.transData
            coords = [single_veh_trj['global_center_x'].iloc[0], single_veh_trj['global_center_y'].iloc[0]]
            if single_veh_trj.iloc[0, 0] == 'ego':
                veh_local_id = 0
                temp_facecolor = 'red'
                temp_alpha = 0.99
                heading_angle = single_veh_trj['heading'].iloc[0] * 180 / np.pi
                tr = mpl.transforms.Affine2D().rotate_deg_around(coords[0], coords[1], heading_angle)
            else:
                # calculate vehicle's local id
                veh_id_match_temp = veh_name_id_match[veh_name_id_match['obj_id'] == signle_veh_id]
                if single_veh_trj['obj_type'].iloc[0] == 'vehicle':
                    # only vehicle has a local id
                    veh_local_id = veh_id_match_temp['local_id'].iloc[0]
                if single_veh_trj['obj_type'].iloc[0] == 'vehicle':
                    temp_facecolor = 'blue'
                elif single_veh_trj['obj_type'].iloc[0] == 'bicycle':
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
                xy=(single_veh_trj['global_center_x'].iloc[0] - 0.5 * veh_length,
                    single_veh_trj['global_center_y'].iloc[0] - 0.5 * veh_width),
                width=veh_length,
                height=veh_width,
                linewidth=0.1,
                facecolor=temp_facecolor,
                edgecolor='black',
                alpha=temp_alpha,
                transform=t))
            # add vehicle local id for only vehicle object
            if single_veh_trj['obj_type'].iloc[0] == 'vehicle':
                temp_text = plt.text(single_veh_trj['global_center_x'].iloc[0],
                                     single_veh_trj['global_center_y'].iloc[0], str(veh_local_id), style='italic',
                                     weight='heavy', ha='center', va='center', color='white', rotation=heading_angle,
                                     size=3)
                temp_text.set_path_effects(
                    [path_effects.Stroke(linewidth=1, foreground='black'), path_effects.Normal()])
    trj_save_name = 'figure_save/temp_top_view_figure/top_view_segment_' + str(seg_id_in) + '_frame_' + str(
        frame_id_in) + '_trajectory.jpg'
    plt.savefig(trj_save_name, dpi=600)
    plt.close('all')

def top_view_video_generation():
    # this function generates one top view video based on top view figures from one segment
    img_array = []
    for num in range(1, len(os.listdir('figure_save/temp_top_view_figure/')) + 1):
        image_filename = 'figure_save/temp_top_view_figure/' + 'top_view_segment_' + str(
            single_seg_id) + '_frame_' + str(num) + '_trajectory.jpg'
        img = cv2.imread(image_filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    video_save_name = 'figure_save/top_view_video/' + 'animation_top_view_segment_' + str(single_seg_id) + '.avi'
    out = cv2.VideoWriter(video_save_name, cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print('top view video made success')
    # after making the video, delete all the frame jpgs
    filelist = glob.glob(os.path.join('figure_save/temp_top_view_figure/', "*.jpg"))
    for f in filelist:
        os.remove(f)



if __name__ == '__main__':

    # ******************** Stage 1: collect lidar object information from original dataset ********************

    segment_name_list = []
    # initialize a list to store the final processed data
    all_segment_all_frame_all_object_info = []

    segment_file = 'D:/Data/WaymoData/training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord'
    segment_id = 1

    segment_name = os.path.basename(segment_file)
    #print(segment_name)   #仅输出文件名
    segment_print = 'Now in segment: ' + str(segment_id)
    print(segment_print)

    single_segment_name_dict = {}
    single_segment_name_dict['segment_id'] = segment_id  # short local id
    single_segment_name_dict['segment_file_name'] = segment_name  # gobal unique long name (file name)
    filename = segment_file
    # read a segment's data
    segment_dataset = tf.data.TFRecordDataset(filename, compression_type='')
    segment_dataset = segment_dataset.apply(tf.data.experimental.ignore_errors())
    # every segment data includes 200 frames
    # our preprocess start from a frame
    frame_label = 0
    #print(len(segment_dataset))

    camera_video_generation_label = 1

    for frame_data in segment_dataset:  #1个segment里面有199帧左右 frame
        frame_label += 1
        print_name = 'Now in frame: ' + str(frame_label)
        print(print_name)

        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(frame_data.numpy()))

        # ---------- camera image generation ----------
        # extract camera images and make a video from the images
        (range_images, camera_projections,
         range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
        # print(frame.context)
        fig = plt.figure(figsize=(8, 4.5))
        for index, image in enumerate(frame.images):
            current_image = show_camera_image(image, frame.camera_labels, [2, 3, index + 1])
        # plt.show()
        image_save_title = 'figure_save/temp_cam_pic/' + 'frame_' + str(frame_label) + '.jpg'
        plt.savefig(image_save_title, dpi=150)
        plt.close()

        # ---------- trajectory extraction ----------
        if frame_label == 1:
            segment_start_time = frame.timestamp_micros

        veh_trj_collect(frame)

        #print(all_segment_all_frame_all_object_info)
        #frame_context_dict = frame_context_update(frame)
        #print(frame)

    # ---------- camera video generation ----------
    if camera_video_generation_label:
        camera_video_generation()
    segment_name_list.append(single_segment_name_dict)

    # save up-to-date results every 100 segments to avoid too big single file
    save_step = 100
    #print(segment_id)
    #if segment_id % save_step == 0:
    segment_name_pd = pd.DataFrame(segment_name_list)
    all_segment_all_frame_all_object_info_pd = pd.DataFrame(all_segment_all_frame_all_object_info)
    final_trj_result_format()
    segment_name_pd.to_csv('data_save/segment_global_name_and_local_id_match.csv', index=False)
    save_name = 'data_save/segment_' + str(segment_id - save_step) + '_' + str(
        segment_id) + '_all_frame_all_object_info.csv'
    all_segment_all_frame_all_object_info_pd.to_csv(save_name, index=False)
    print(save_name)
    # must restore contanier's state, or the memory could spill
    all_segment_all_frame_all_object_info = []

# ******************** Stage 2: visualization of lidar information ********************

# ---------- process calulated trajectories from the csv lidar information file ----------
local_veh_id_generation_label = 1  # a local vehicle ID will make each object more identifiable
if local_veh_id_generation_label:
    # ---------- process calulated trajectories from the csv file ----------
    all_segment_all_vehicle_object_id_and_local_id_match = []
    save_step = 100
    save_segment_id_start = 0
    total_steps = 2
    for i in range(1, total_steps):   #total_steps在做什么？
        #read_name = 'data_save/' + 'segment_' + str(save_segment_id_start) + '_' + str(save_segment_id_start + save_step) + '_all_frame_all_object_info.csv'

        read_name = 'data_save/segment_-99_1_all_frame_all_object_info.csv'  #!!!需要修改


        temp_trj = pd.read_csv(read_name, header=0)
        all_segment_id = pd.unique(temp_trj['segment_id'])
        for single_seg_id in all_segment_id:
            segment_print = 'Now in segment (generating local id): ' + str(single_seg_id)
            print(segment_print)
            l_seg_id = temp_trj['segment_id'] == single_seg_id
            seg_trj = temp_trj[l_seg_id]
            all_segment_veh_id = pd.unique(seg_trj['obj_id'])
            #print(all_segment_veh_id)
            num_label = 1
            for single_veh_id in all_segment_veh_id:
                # print(single_veh_id)
                l_obj_id = seg_trj['obj_id'] == single_veh_id
                single_obj_info = seg_trj.loc[l_obj_id, :]
                if single_obj_info['obj_type'].iloc[0] == 'vehicle':
                    single_match = {}
                    single_match['segment_id'] = single_seg_id
                    if single_veh_id == 'ego':
                        # note that AV's local id is always 0
                        continue
                    else:
                        single_match['obj_id'] = single_veh_id
                        single_match['local_id'] = num_label
                        num_label += 1
                        all_segment_all_vehicle_object_id_and_local_id_match.append(single_match)
        save_segment_id_start += save_step
    # save the vehicle object id & local id match data
    all_segment_all_vehicle_object_id_and_local_id_match_pd = pd.DataFrame(
        all_segment_all_vehicle_object_id_and_local_id_match)
    all_segment_all_vehicle_object_id_and_local_id_match_pd.to_csv(
            'data_save/all_segment_all_vehicle_object_id_and_local_id_match.csv', index=False)


    # # to determine if the top view video should be generated in this run
    top_view_video_generation_label = 1
    if top_view_video_generation_label:
        save_step = 100
        save_segment_id_start = 0
        total_steps = 2
        for i in range(1, total_steps):
            #trj_read_name = 'data_save/' + 'segment_' + str(save_segment_id_start) + '_' + str(save_segment_id_start + save_step) + '_all_frame_all_object_info.csv'

            trj_read_name = 'data_save/segment_-99_1_all_frame_all_object_info.csv'#!!!需要修改


            temp_trj = pd.read_csv(trj_read_name, header=0,
                                   dtype={'global_center_x': np.float64, 'global_center_y': np.float64})
            # this match file has three columns [segment_id, obj_id, local_id]

            veh_name_id_match = pd.read_csv('data_save/all_segment_all_vehicle_object_id_and_local_id_match.csv',
                                                header=0)
            all_segment_id = pd.unique(temp_trj['segment_id'])
            # delete previous frame jpgs (might or might not exist)
            filelist = glob.glob(os.path.join('figure_save/temp_top_view_figure/', "*.jpg"))
            for f in filelist:
                os.remove(f)
            for single_seg_id in all_segment_id:
                if single_seg_id >= 1:
                    segment_print = 'Top view video now in segment: ' + str(single_seg_id)
                    print(segment_print)
                    l_seg_id = temp_trj['segment_id'] == single_seg_id
                    seg_trj = temp_trj[l_seg_id]
                    top_view_trj = seg_trj[
                        ['obj_id', 'obj_type', 'frame_label', 'local_time_stamp', 'global_center_x', 'global_center_y',
                         'width', 'length', 'heading']]
                    total_frame_num = seg_trj['frame_label'].max()
                    for frame_id in range(1, total_frame_num + 1):
                        if frame_id == 1:
                            current_frame_trj = top_view_trj[top_view_trj['frame_label'] == frame_id]
                        plot_top_view_ani_with_lidar_label(top_view_trj, single_seg_id, frame_id)
                    # ---------- video generation ----------
                    top_view_video_generation()
            save_segment_id_start += save_step


