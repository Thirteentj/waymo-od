import os
# import tensorflow as tf
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


def camera_video_generation():
    img_array = []
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


def frame_context_update(frame_in):
    # collect environment context in this frame
    frame_context_dict = {}
    frame_context_dict['segment_id'] = segment_id
    frame_context_dict['frame_label'] = frame_label
    frame_context_dict['time_of_day'] = frame_in.context.stats.time_of_day
    frame_context_dict['location'] = frame_in.context.stats.location
    frame_context_dict['weather'] = frame_in.context.stats.weather
    for count in frame_in.context.stats.laser_object_counts:
        if count.type != 1:  # note that 1 means vehicle object
            continue
        frame_context_dict['laser_veh_count'] = count.count
    return frame_context_dict


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
        np.array([row_dict['local_center_x'], row_dict['local_center_y'], row_dict['local_center_z'], 1]), (4, 1))
    # print(local_pos_matrix)
    label_global_pos = np.matmul(trans_matrix, local_pos_matrix)
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


def veh_trj_collect(frame_in):
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
    for lidar_label in frame_in.laser_labels:
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


def plot_top_view_ani_with_lidar_label(trj_in, seg_id_in, frame_id_in):
    # this function plots one single frame of the top view video
    # trj_in is a pandas with three columns(obj_id, frame_label, local_time_stamp, global_center_x, global_center_y, length, width, heading)
    # trj_in is all the trajectories within one segment
    # seg_id_in is the current segment id
    trj_in['global_center_x'] = trj_in['global_center_x'] - trj_in['global_center_x'].min()   #进行坐标原点的一个平移
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
                tr = mpl.transforms.Affine2D().rotate_deg_around(coords[0], coords[1], heading_angle)  #对车辆按照航向角进行旋转
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


def cumulated_dis_cal(coord_series_in, segment_id_in, veh_id_in, start_time_in):
    # this function calculate the cumulated distance based on the given  global coordinates,
    # input coord_series_in: ['global_center_x', 'global_center_y', 'speed_x', 'speed_y']
    # output coord_series_in: ['global_center_x', 'global_center_y', 'speed_x', 'speed_y', 'cumu_dis', 'speed', 'accer', 'filter_cumu_dis',
    # 'filter_speed', 'filter_accer', 'speed_based_cumu_dis', 'speed_based_speed', 'speed_based_accer', 'speed_based_filter_cumu_dis',
    # 'speed_based_filter_speed', 'speed_based_accer']
    coord_series_in.reset_index(drop=True, inplace=True)
    coord_series_in.loc[:, 'cumu_dis'] = float(0)
    coord_series_in.loc[:, 'speed'] = float(0)
    coord_series_in.loc[:, 'accer'] = float(0)
    coord_series_in.loc[:, 'speed_based_cumu_dis'] = float(0)
    coord_series_in.loc[:, 'speed_based_speed'] = float(0)
    coord_series_in.loc[:, 'speed_based_accer'] = float(0)
    coord_series_in.loc[:, 'speed_based_jerk'] = float(0)
    # calculate distance for position based method, and speed for speed based method
    for i in range(1, len(coord_series_in['global_center_x'])):
        pre_x = coord_series_in['global_center_x'].iloc[i - 1]
        pre_y = coord_series_in['global_center_y'].iloc[i - 1]
        post_x = coord_series_in['global_center_x'].iloc[i]
        post_y = coord_series_in['global_center_y'].iloc[i]
        single_dis = math.sqrt((post_x - pre_x) ** 2 + (post_y - pre_y) ** 2)
        coord_series_in.loc[i, 'cumu_dis'] = coord_series_in.loc[i - 1, 'cumu_dis'] + single_dis
    for i in range(len(coord_series_in['global_center_x'])):
        speed_based_speed = math.sqrt((coord_series_in.at[i, 'speed_x']) ** 2 + (coord_series_in.at[i, 'speed_y']) ** 2)
        coord_series_in.loc[i, 'speed_based_speed'] = speed_based_speed
    # calculate speed and acceleration position based method, distance and aceleration for speed based method
    coord_series_in = update_speed_and_accer(coord_series_in, 0)
    coord_series_in = speed_based_update_distance_and_accer(coord_series_in)
    # trajectory correctness
    # initialize filter_value
    coord_series_in.loc[:, 'filter_cumu_dis'] = coord_series_in.loc[:, 'cumu_dis'].to_numpy()
    coord_series_in.loc[:, 'filter_speed'] = coord_series_in.loc[:, 'speed'].to_numpy()
    coord_series_in.loc[:, 'filter_accer'] = coord_series_in.loc[:, 'accer'].to_numpy()
    coord_series_in.loc[:, 'filter_jerk'] = 0
    coord_series_in = trajectory_correctness(coord_series_in, segment_id_in, veh_id_in, start_time_in)
    return coord_series_in


def speed_based_update_distance_and_accer(series_in):
    # this function calculate the distance, acceleration and jerk based on speed (for speed-based data)
    # series_in is the same format as  coord_series_in
    # output is series_in with updated speed and accer
    current_cumu_dis = 'speed_based_cumu_dis'
    current_speed = 'speed_based_speed'
    current_accer = 'speed_based_accer'
    for i in range(1, len(series_in['global_center_x'])):
        if i == 1:
            series_in.loc[0, current_cumu_dis] = 0
            series_in.loc[i, current_cumu_dis] = series_in.loc[i - 1, current_cumu_dis] + (
                    series_in.loc[i, current_speed] + series_in.loc[i - 1, current_speed]) * 0.5 * 0.1
        else:
            series_in.loc[i, current_cumu_dis] = series_in.loc[i - 1, current_cumu_dis] + (
                    series_in.loc[i, current_speed] + series_in.loc[i - 1, current_speed]) * 0.5 * 0.1
    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_accer] = float(
                series_in.at[i + 2, current_speed] - series_in.at[i, current_speed]) / (float(0.2))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_accer] = float(
                series_in.at[i, current_speed] - series_in.at[i - 2, current_speed]) / (float(0.2))
        else:
            series_in.at[i, current_accer] = float(
                series_in.at[i + 1, current_speed] - series_in.at[i - 1, current_speed]) / (float(0.2))
    current_jerk = 'speed_based_jerk'
    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_jerk] = float(
                series_in.at[i + 2, current_accer] - series_in.at[i, current_accer]) / (float(0.2))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_jerk] = float(
                series_in.at[i, current_accer] - series_in.at[i - 2, current_accer]) / (float(0.2))
        else:
            series_in.at[i, current_jerk] = float(
                series_in.at[i + 1, current_accer] - series_in.at[i - 1, current_accer]) / (float(0.2))
    return series_in


def update_speed_and_accer(series_in, filter_label):
    # this function calculate the speed, accelearation, jerk based on position
    # series_in is the same format as  coord_series_in
    # output is series_in with updated speed and accer
    if filter_label == 1:
        current_cumu_dis = 'filter_cumu_dis'
        current_speed = 'filter_speed'
        current_accer = 'filter_accer'
    elif filter_label == 0:
        current_cumu_dis = 'cumu_dis'
        current_speed = 'speed'
        current_accer = 'accer'
        current_jerk = 'jerk'
    else:
        # label should be 2
        current_cumu_dis = 'remove_outlier_cumu_dis'
        current_speed = 'remove_outlier_speed'
        current_accer = 'remove_outlier_accer'
        current_jerk = 'remove_outlier_jerk'
    # calculate speed
    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_speed] = float(
                series_in.at[i + 2, current_cumu_dis] - series_in.at[i, current_cumu_dis]) / (float(0.2))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_speed] = float(
                series_in.at[i, current_cumu_dis] - series_in.at[i - 2, current_cumu_dis]) / (float(0.2))
        else:
            series_in.at[i, current_speed] = float(
                series_in.at[i + 1, current_cumu_dis] - series_in.at[i - 1, current_cumu_dis]) / (float(0.2))
    # calculate accerleration
    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_accer] = float(
                series_in.at[i + 2, current_speed] - series_in.at[i, current_speed]) / (float(0.2))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_accer] = float(
                series_in.at[i, current_speed] - series_in.at[i - 2, current_speed]) / (float(0.2))
        else:
            series_in.at[i, current_accer] = float(
                series_in.at[i + 1, current_speed] - series_in.at[i - 1, current_speed]) / (float(0.2))
    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_jerk] = float(
                series_in.at[i + 2, current_accer] - series_in.at[i, current_accer]) / (float(0.2))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_jerk] = float(
                series_in.at[i, current_accer] - series_in.at[i - 2, current_accer]) / (float(0.2))
        else:
            series_in.at[i, current_jerk] = float(
                series_in.at[i + 1, current_accer] - series_in.at[i - 1, current_accer]) / (float(0.2))
    return series_in


def plot_outlier_adjacent_trj(series_in, outlier_pos_in, first_pos_in, last_pos_in, segment_id_in, veh_id_in, start_time_in, comparison_label):
    # plot the adjacent trajectory of the outlier (20 points)
    outlier_time = round(start_time_in + outlier_pos_in * 0.1, 1)
    included_index = np.arange(first_pos_in, last_pos_in + 1, dtype=int)
    outlier_trj = series_in.loc[included_index, :]
    outlier_trj.loc[:, 'local_time'] = np.array(included_index) * 0.1 + start_time_in
    plt.subplot(3, 1, 1)
    plt.plot(outlier_trj['local_time'], outlier_trj['cumu_dis'], '-*k', linewidth=0.25, label='Original', markersize=1.5)
    if comparison_label == 1:
        plt.plot(outlier_trj['local_time'], outlier_trj['remove_outlier_cumu_dis'], '-m', linewidth=0.25, label='Outliers Removed')
        plt.legend(prop={'size': 6})
        trj_title = 'Segment ' + str(int(segment_id_in)) + ' Vehicle' + str(
            int(veh_id_in)) + ' Outlier at Time ' + str(outlier_time) + ' Removing'
    else:
        trj_title = 'Segment ' + str(int(segment_id_in)) + ' Vehicle' + str(
            int(veh_id_in)) + ' Outlier at Time ' + str(outlier_time) + ' Pattern'
    plt.ylabel('Position (m)')
    plt.title(trj_title)
    plt.subplot(3, 1, 2)
    plt.plot(outlier_trj['local_time'], outlier_trj['speed'], '-*k', linewidth=0.5, label='Original', markersize=1.5)
    if comparison_label == 1:
        plt.plot(outlier_trj['local_time'], outlier_trj['remove_outlier_speed'], '-m', linewidth=0.5, label='Outliers Removed')
        plt.legend(prop={'size': 6})
    plt.ylabel('Speed (m/s)')
    plt.ylim([0, 35])
    plt.subplot(3, 1, 3)
    plt.plot(outlier_trj['local_time'], outlier_trj['accer'], '-*k', linewidth=0.5, label='Original', markersize=1.5)
    if comparison_label == 1:
        plt.plot(outlier_trj['local_time'], outlier_trj['remove_outlier_accer'], '-m', linewidth=0.5, label='Outliers Removed')
        plt.legend(prop={'size': 6})
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s2)')
    plt.ylim([-15, 15])
    trj_save_title = 'figure_save/trajectory_process/outlier_pattern_and_removing/' + trj_title + '.png'
    plt.savefig(trj_save_title, dpi=600)
    plt.close('all')


def outlier_removing_optimization_model(initial_state_in, last_state_in, num_points_in):
    # note that the num_points_in includes the first and last points
    # if the total number of interpolated points is n, then  num_points_in = n + 2
    # time interval is 0.1 second
    # total number of time steps
    max_acc = 5
    min_acc = -8
    total_steps = num_points_in
    first_pos_in = initial_state_in[0]
    first_speed_in = initial_state_in[1]
    first_acc_in = initial_state_in[2]
    last_pos_in = last_state_in[0]
    last_speed_in = last_state_in[1]
    last_acc_in = last_state_in[2]
    # time interval in each step
    time_interval = 0.1
    # model = GEKKO() # Initialize gekko
    model = GEKKO(remote=False)  # Initialize gekko
    # Use IPOPT solver (default)
    model.options.SOLVER = 3
    model.options.SCALING = 2
    # Initialize variables
    acc = [None] * total_steps  # simulated acceleration
    velocity = [None] * total_steps  # simulated velocity
    pos = [None] * total_steps  # simulated position
    for i in range(total_steps):
        pos[i] = model.Var()
        velocity[i] = model.Var()
        velocity[i].lower = 0
        acc[i] = model.Var(lb=min_acc, ub=max_acc)
    min_sim_acc = model.Var()
    max_sim_acc = model.Var()
    model.Equation(pos[0] == first_pos_in)
    model.Equation(velocity[0] == first_speed_in)
    model.Equation(acc[0] == first_acc_in)
    model.Equation(pos[total_steps - 1] == last_pos_in)
    model.Equation(velocity[total_steps - 1] == last_speed_in)
    model.Equation(acc[total_steps - 1] == last_acc_in)
    for i in range(total_steps):
        if 1 <= i <= total_steps - 1:
            model.Equation(velocity[i] == velocity[i - 1] + acc[i - 1] * time_interval)
            model.Equation(pos[i] == pos[i - 1] + 0.5 * (velocity[i] + velocity[i - 1]) * time_interval)
    for i in range(total_steps):
        model.Equation(min_sim_acc <= acc[i])
        model.Equation(max_sim_acc >= acc[i])
    # objective function: minimize the difference between max_sim_acc and min_sim_acc
    model.Obj(max_sim_acc - min_sim_acc)
    # model.options.IMODE = 2  # Steady state optimization
    model.options.MAX_MEMORY = 5
    model.solve(disp=False)
    # solve_time = model.options.SOLVETIME
    # extract values from Gekko type variables
    acc_value = np.zeros(total_steps)
    velocity_value = np.zeros(total_steps)
    pos_value = np.zeros(total_steps)
    for i in range(total_steps):
        acc_value[i] = acc[i].value[0]
        velocity_value[i] = velocity[i].value[0]
        pos_value[i] = pos[i].value[0]
    return pos_value, velocity_value, acc_value


def optimization_based_outlier_removing(series_in, first_pos_in, last_pos_in, min_acc_in, max_acc_in):
    # given the position of the outlier, optimize its vicinity's trajectory
    first_point_pos = first_pos_in
    last_point_pos = last_pos_in
    first_point_cumu_dis = series_in.at[first_point_pos, 'remove_outlier_cumu_dis']
    first_point_speed = series_in.at[first_point_pos, 'remove_outlier_speed']
    if series_in.at[first_point_pos, 'remove_outlier_accer'] <= min_acc_in:
        first_point_acc = min_acc_in
    elif series_in.at[first_point_pos, 'remove_outlier_accer'] >= max_acc_in:
        first_point_acc = max_acc_in
    else:
        first_point_acc = series_in.at[first_point_pos, 'remove_outlier_accer']
    first_point_state = [first_point_cumu_dis, first_point_speed, first_point_acc]
    last_point_cumu_dis = series_in.at[last_point_pos, 'remove_outlier_cumu_dis']
    last_point_speed = series_in.at[last_point_pos, 'remove_outlier_speed']
    if series_in.at[last_point_pos, 'remove_outlier_accer'] <= min_acc_in:
        last_point_acc = min_acc_in
    elif series_in.at[last_point_pos, 'remove_outlier_accer'] >= max_acc_in:
        last_point_acc = max_acc_in
    else:
        last_point_acc = series_in.at[last_point_pos, 'remove_outlier_accer']
    last_point_state = [last_point_cumu_dis, last_point_speed, last_point_acc]
    actual_total_related_points = last_point_pos - first_point_pos + 1
    pos_result, speed_result, acc_result = outlier_removing_optimization_model(first_point_state, last_point_state, actual_total_related_points)
    series_in.loc[first_point_pos:last_point_pos, 'remove_outlier_cumu_dis'] = pos_result
    series_in = update_speed_and_accer(series_in, 2)
    return series_in


def wavefilter(data):
    # We will use the Daubechies(6) wavelet
    daubechies_num = 6
    wname = "db" + str(daubechies_num)
    datalength = data.shape[0]
    max_level = pywt.dwt_max_level(datalength, wname)
    print('maximun level is: %s' % max_level)
    # Initialize the container for the filtered data
    # Decompose the signal
    # coeff[0] is approximation coeffs, coeffs[1] is nth level detail coeff, coeff[-1] is first level detail coeffs
    coeffs = pywt.wavedec(data, wname, mode='smooth', level=max_level)
    # thresholding
    for j in range(max_level):
        coeffs[-j - 1] = np.zeros_like(coeffs[-j - 1])
    # Reconstruct the signal and save it
    filter_data = pywt.waverec(coeffs, wname, mode='smooth')
    fdata = filter_data[0:datalength]
    return fdata


def wavelet_filter(series_in):
    remove_outlier_speed_signal = series_in.loc[:, 'remove_outlier_speed'].to_numpy()
    wavelet_filter_speed = wavefilter(remove_outlier_speed_signal)
    series_in.loc[:, 'wavelet_filter_speed'] = wavelet_filter_speed
    series_in.loc[:, 'wavelet_filter_cumu_dis'] = None
    series_in.loc[:, 'wavelet_filter_accer'] = None
    series_in.loc[:, 'wavelet_filter_jerk'] = None
    # update cumulative distance
    for i in range(len(series_in['global_center_x'])):
        if i == 0:
            # start from the filtered value
            series_in.loc[i, 'wavelet_filter_cumu_dis'] = 0  # initial pos should be 0
        else:
            series_in.loc[i, 'wavelet_filter_cumu_dis'] = series_in.loc[i - 1, 'wavelet_filter_cumu_dis'] + (
                    series_in.loc[i - 1, 'wavelet_filter_speed'] + series_in.loc[i, 'wavelet_filter_speed']) * 0.5 * 0.1
    # update acceleration
    current_speed = 'wavelet_filter_speed'
    current_accer = 'wavelet_filter_accer'
    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_accer] = float(
                series_in.at[i + 2, current_speed] - series_in.at[i, current_speed]) / (float(0.2))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_accer] = float(
                series_in.at[i, current_speed] - series_in.at[i - 2, current_speed]) / (float(0.2))
        else:
            series_in.at[i, current_accer] = float(
                series_in.at[i + 1, current_speed] - series_in.at[i - 1, current_speed]) / (float(0.2))
    current_jerk = 'wavelet_filter_jerk'
    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_jerk] = float(
                series_in.at[i + 2, current_accer] - series_in.at[i, current_accer]) / (float(0.2))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_jerk] = float(
                series_in.at[i, current_accer] - series_in.at[i - 2, current_accer]) / (float(0.2))
        else:
            series_in.at[i, current_jerk] = float(
                series_in.at[i + 1, current_accer] - series_in.at[i - 1, current_accer]) / (float(0.2))
    return series_in


def trajectory_correctness(coord_series_in, segment_id_in, veh_id_in, start_time_in):
    # this function remove outliers and filter the trajectory
    # input coord_series_in: ['global_center_x', 'global_center_y', 'cumu_dis', 'speed', 'accer']
    # output coord_series_in: ['global_center_x', 'global_center_y', 'cumu_dis', 'speed', 'accer', 'filter_cumu_dis', 'filter_speed', 'filter_accer']
    minimum_accer = -8
    maximum_accer = 5
    reference_points_num = 20
    coord_series_in.reset_index(inplace=True, drop=True)
    global all_outlier_record
    # remove outliers in acceleration, note that cubic spline interpolation is implemented on distance
    # initialize remove outlier results
    coord_series_in.loc[:, 'remove_outlier_cumu_dis'] = coord_series_in.loc[:, 'cumu_dis']
    coord_series_in.loc[:, 'remove_outlier_speed'] = coord_series_in.loc[:, 'speed']
    coord_series_in.loc[:, 'remove_outlier_accer'] = coord_series_in.loc[:, 'accer']
    # removing outliers should be conducted multiple times until there is no outlier
    outlier_label = 1
    while outlier_label:
        outlier_label = 0
        for m in range(len(coord_series_in['global_center_x'])):
            if coord_series_in.at[m, 'remove_outlier_accer'] >= maximum_accer or coord_series_in.at[m, 'remove_outlier_accer'] <= minimum_accer:
                print('Outlier info: Current segment: %s, vehicle id: %s, time: %s, position: %s' % (
                    segment_id_in, veh_id_in, round(m * 0.1 + start_time_in, 1), m))
                single_outlier_record = pd.DataFrame(np.zeros((1, 3)), columns=['segment_id', 'local_veh_id', 'outlier_time'])
                single_outlier_record.loc[0, 'segment_id'] = segment_id_in
                single_outlier_record.loc[0, 'local_veh_id'] = veh_id_in
                single_outlier_record.loc[0, 'outlier_time'] = start_time_in + 0.1 * m
                all_outlier_record = all_outlier_record.append(single_outlier_record)
                total_related_points = 20
                first_point_pos = int(max(0, m - total_related_points / 2))
                last_point_pos = int(min(len(coord_series_in.loc[:, 'remove_outlier_accer']) - 1, m + total_related_points / 2))
                if first_point_pos == 0:
                    last_point_pos = first_point_pos + total_related_points
                if last_point_pos == len(coord_series_in.loc[:, 'remove_outlier_accer']) - 1:
                    first_point_pos = last_point_pos - total_related_points
                plot_outlier_adjacent_trj(coord_series_in, m, first_point_pos, last_point_pos, segment_id_in, veh_id_in, start_time_in, 0)
                # the following pairs may not have feasible solutions during outlier removal
                if segment_id_in == 191 and veh_id_in == 6:
                    pass
                elif segment_id_in == 270 and veh_id_in == 4:
                    pass
                elif segment_id_in == 276 and veh_id_in == 2:
                    pass
                elif segment_id_in == 320 and veh_id_in == 1:
                    pass
                elif segment_id_in == 406 and veh_id_in == 25:
                    pass
                elif segment_id_in == 449 and veh_id_in == 41:
                    pass
                elif segment_id_in == 450 and veh_id_in == 15:
                    pass
                elif segment_id_in == 676 and veh_id_in == 15:
                    pass
                elif segment_id_in == 769 and veh_id_in == 50:
                    pass
                elif segment_id_in == 916 and veh_id_in == 4:
                    pass
                elif segment_id_in == 968 and veh_id_in == 18:
                    pass
                else:
                    coord_series_in = optimization_based_outlier_removing(coord_series_in, first_point_pos, last_point_pos, minimum_accer,
                                                                          maximum_accer)
                    plot_outlier_adjacent_trj(coord_series_in, m, first_point_pos, last_point_pos, segment_id_in, veh_id_in, start_time_in, 1)
                outlier_label = 0  # outlier still exsit in this loop
    # implement wavelet filter after removing outliers
    coord_series_in = wavelet_filter(coord_series_in)
    # set the final filter results to the wavelet filte results
    coord_series_in.loc[:, 'filter_cumu_dis'] = coord_series_in.loc[:, 'wavelet_filter_cumu_dis'].to_numpy()
    coord_series_in.loc[:, 'filter_speed'] = coord_series_in.loc[:, 'wavelet_filter_speed'].to_numpy()
    coord_series_in.loc[:, 'filter_accer'] = coord_series_in.loc[:, 'wavelet_filter_accer'].to_numpy()
    coord_series_in.loc[:, 'filter_jerk'] = coord_series_in.loc[:, 'wavelet_filter_jerk'].to_numpy()
    return coord_series_in


def before_and_after_remove_outlier_plot(trj_in):
    current_seg_id = trj_in['segment_id'].iloc[0]
    follower_id_in = trj_in['local_veh_id'].iloc[0]
    if len(all_outlier_record) > 0:
        current_seg_outlier_record = all_outlier_record.loc[
                                     all_outlier_record['segment_id'] == current_seg_id, :]
        current_seg_outlier_record_local_veh_id = current_seg_outlier_record.loc[:, 'local_veh_id'].to_numpy().astype(np.int32)
    else:
        current_seg_outlier_record_local_veh_id = []
    if int(follower_id_in) in current_seg_outlier_record_local_veh_id:
        plt.subplot(3, 1, 1)
        plt.plot(trj_in['local_time'], trj_in['position'], '--k', linewidth=0.25, label='Original')
        plt.plot(trj_in['local_time'], trj_in['remove_outlier_pos'], '-m', linewidth=0.25, label='Outliers Removed')
        plt.ylabel('Position (m)')
        plt.legend(prop={'size': 6})
        trj_title = 'Segment ' + str(int(current_seg_id)) + ' Vehicle' + str(
            int(follower_id_in)) + ' Before and After Removing Outliers'
        plt.title(trj_title)
        plt.subplot(3, 1, 2)
        plt.plot(trj_in['local_time'], trj_in['speed'], '--k', linewidth=0.5, label='Original')
        plt.plot(trj_in['local_time'], trj_in['remove_outlier_speed'], '-m', linewidth=0.5, label='Outliers Removed')
        plt.ylabel('Speed (m/s)')
        plt.legend(prop={'size': 6})
        plt.ylim([0, 35])
        plt.subplot(3, 1, 3)
        plt.plot(trj_in['local_time'], trj_in['accer'], '--k', linewidth=0.5, label='Original')
        plt.plot(trj_in['local_time'], trj_in['remove_outlier_accer'], '-m', linewidth=0.5, label='Outliers Removed')
        plt.legend(prop={'size': 6})
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s2)')
        plt.ylim([-15, 15])
        trj_save_title = 'figure_save/trajectory_process/before_and_after_remove_outlier_plot/' + trj_title + '.png'
        plt.savefig(trj_save_title, dpi=600)
        plt.close('all')


def before_and_after_filtering_plot(trj_in):
    current_seg_id = trj_in['segment_id'].iloc[0]
    follower_id_in = trj_in['local_veh_id'].iloc[0]
    if len(all_outlier_record) > 0:
        current_seg_outlier_record = all_outlier_record.loc[
                                     all_outlier_record['segment_id'] == current_seg_id, :]
        current_seg_outlier_record_local_veh_id = current_seg_outlier_record.loc[:, 'local_veh_id'].to_numpy().astype(np.int32)
    else:
        current_seg_outlier_record_local_veh_id = []
    if int(follower_id_in) in current_seg_outlier_record_local_veh_id:
        plt.subplot(3, 1, 1)
        plt.plot(trj_in['local_time'], trj_in['position'], '--k', linewidth=0.25, label='Original')
        plt.plot(trj_in['local_time'], trj_in['remove_outlier_pos'], '-m', linewidth=0.25, label='Outliers Removed')
        plt.plot(trj_in['local_time'], trj_in['wavelet_filter_pos'], '-*g', linewidth=0.25, label='Outliers Removed + Filtering', markersize=0.5)
        plt.ylabel('Position (m)')
        plt.legend(prop={'size': 6})
        trj_title = 'Segment ' + str(int(current_seg_id)) + ' Vehicle' + str(
            int(follower_id_in)) + ' Before and After Filtering'
        plt.title(trj_title)
        plt.subplot(3, 1, 2)
        plt.plot(trj_in['local_time'], trj_in['speed'], '--k', linewidth=0.25, label='Original')
        plt.plot(trj_in['local_time'], trj_in['remove_outlier_speed'], '-m', linewidth=0.25, label='Outliers Removed')
        plt.plot(trj_in['local_time'], trj_in['wavelet_filter_speed'], '-*g', linewidth=0.25, label='Outliers Removed + Filtering', markersize=0.5)
        plt.ylabel('Speed (m/s)')
        plt.legend(prop={'size': 6})
        plt.ylim([0, 35])
        plt.subplot(3, 1, 3)
        plt.plot(trj_in['local_time'], trj_in['accer'], '--k', linewidth=0.25, label='Original')
        plt.plot(trj_in['local_time'], trj_in['remove_outlier_accer'], '-m', linewidth=0.25, label='Outliers Removed')
        plt.plot(trj_in['local_time'], trj_in['wavelet_filter_accer'], '-*g', linewidth=0.25, label='Outliers Removed + Filtering', markersize=0.5)
        plt.legend(prop={'size': 6})
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s2)')
        plt.ylim([-15, 15])
        trj_save_title = 'figure_save/trajectory_process/before_and_after_filtering_plot/' + trj_title + '.png'
        plt.savefig(trj_save_title, dpi=600)
        plt.close('all')


def pair_cf_coord_cal(leader_id, leader_trj_in, follower_id, follower_trj_in, av_label):
    # convert 2-d coordinates to 1-d longitudinal coordinates
    # note that the leader and follower interacts with each other
    # av_label is to determine whether av is leader or follower (0 for follower, 1 for leader, 2 for non-av pair)
    global all_seg_paired_cf_trj_final
    global all_seg_paired_cf_trj_with_comparison
    # extract mutual cf trajectory
    min_local_time = max(leader_trj_in['local_time_stamp'].min(), follower_trj_in['local_time_stamp'].min())
    max_local_time = min(leader_trj_in['local_time_stamp'].max(), follower_trj_in['local_time_stamp'].max())
    leader_trj_in = leader_trj_in.loc[leader_trj_in['local_time_stamp'] >= min_local_time, :]
    leader_trj_in = leader_trj_in.loc[leader_trj_in['local_time_stamp'] <= max_local_time, :]
    follower_trj_in = follower_trj_in.loc[follower_trj_in['local_time_stamp'] >= min_local_time, :]
    follower_trj_in = follower_trj_in.loc[follower_trj_in['local_time_stamp'] <= max_local_time, :]
    # sort the trj
    leader_trj_in = leader_trj_in.sort_values(['local_time_stamp'])
    follower_trj_in = follower_trj_in.sort_values(['local_time_stamp'])
    # initialize output format
    out_leader_trj = pd.DataFrame(leader_trj_in[['segment_id', 'veh_id', 'length', 'local_time_stamp']].to_numpy(),
                                  columns=['segment_id', 'local_veh_id', 'length', 'local_time'])
    out_leader_trj.loc[:, 'follower_id'] = follower_id
    out_leader_trj.loc[:, 'leader_id'] = leader_id
    out_follower_trj = pd.DataFrame(follower_trj_in[['segment_id', 'veh_id', 'length', 'local_time_stamp']].to_numpy(),
                                    columns=['segment_id', 'local_veh_id', 'length', 'local_time'])
    out_follower_trj.loc[:, 'follower_id'] = follower_id
    out_follower_trj.loc[:, 'leader_id'] = leader_id
    # calculate coordinates of leader and follower
    temp_current_segment_id = out_follower_trj['segment_id'].iloc[0]
    temp_start_time = out_follower_trj['local_time'].iloc[0]
    leader_cumu_dis = cumulated_dis_cal(
        leader_trj_in.loc[:, ['global_center_x', 'global_center_y', 'speed_x', 'speed_y']], temp_current_segment_id, leader_id, temp_start_time)
    follower_cumu_dis = cumulated_dis_cal(
        follower_trj_in.loc[:, ['global_center_x', 'global_center_y', 'speed_x', 'speed_y']], temp_current_segment_id, follower_id, temp_start_time)
    # calculate initial distance
    pre_x_1 = leader_trj_in['global_center_x'].iloc[0]
    pre_y_1 = leader_trj_in['global_center_y'].iloc[0]
    post_x_1 = follower_trj_in['global_center_x'].iloc[0]
    post_y_1 = follower_trj_in['global_center_y'].iloc[0]
    initial_dis = math.sqrt((post_x_1 - pre_x_1) ** 2 + (post_y_1 - pre_y_1) ** 2)
    # create position, speed, and acceleration data
    # follower's position always start from 0
    # position based
    out_follower_trj.loc[:, 'position'] = follower_cumu_dis['cumu_dis'].to_numpy()
    out_follower_trj.loc[:, 'remove_outlier_pos'] = follower_cumu_dis['remove_outlier_cumu_dis'].to_numpy()
    out_follower_trj.loc[:, 'filter_pos'] = follower_cumu_dis['filter_cumu_dis'].to_numpy()
    out_follower_trj.loc[:, 'wavelet_filter_pos'] = follower_cumu_dis['wavelet_filter_cumu_dis'].to_numpy()
    out_follower_trj.loc[:, 'speed'] = follower_cumu_dis['speed'].to_numpy()
    out_follower_trj.loc[:, 'remove_outlier_speed'] = follower_cumu_dis['remove_outlier_speed'].to_numpy()
    out_follower_trj.loc[:, 'filter_speed'] = follower_cumu_dis['filter_speed'].to_numpy()
    out_follower_trj.loc[:, 'wavelet_filter_speed'] = follower_cumu_dis['wavelet_filter_speed'].to_numpy()
    out_follower_trj.loc[:, 'accer'] = follower_cumu_dis['accer'].to_numpy()
    out_follower_trj.loc[:, 'remove_outlier_accer'] = follower_cumu_dis['remove_outlier_accer'].to_numpy()
    out_follower_trj.loc[:, 'filter_accer'] = follower_cumu_dis['filter_accer'].to_numpy()
    out_follower_trj.loc[:, 'wavelet_filter_accer'] = follower_cumu_dis['wavelet_filter_accer'].to_numpy()
    out_follower_trj.loc[:, 'jerk'] = follower_cumu_dis['jerk'].to_numpy()
    out_follower_trj.loc[:, 'filter_jerk'] = follower_cumu_dis['filter_jerk'].to_numpy()
    out_follower_trj.loc[:, 'wavelet_filter_jerk'] = follower_cumu_dis['wavelet_filter_jerk'].to_numpy()
    out_leader_trj.loc[:, 'position'] = leader_cumu_dis['cumu_dis'].to_numpy() + initial_dis
    out_leader_trj.loc[:, 'remove_outlier_pos'] = leader_cumu_dis['remove_outlier_cumu_dis'].to_numpy() + initial_dis
    out_leader_trj.loc[:, 'filter_pos'] = leader_cumu_dis['filter_cumu_dis'].to_numpy() + initial_dis
    out_leader_trj.loc[:, 'wavelet_filter_pos'] = leader_cumu_dis['wavelet_filter_cumu_dis'].to_numpy() + initial_dis
    out_leader_trj.loc[:, 'speed'] = leader_cumu_dis['speed'].to_numpy()
    out_leader_trj.loc[:, 'remove_outlier_speed'] = leader_cumu_dis['remove_outlier_speed'].to_numpy()
    out_leader_trj.loc[:, 'filter_speed'] = leader_cumu_dis['filter_speed'].to_numpy()
    out_leader_trj.loc[:, 'wavelet_filter_speed'] = leader_cumu_dis['wavelet_filter_speed'].to_numpy()
    out_leader_trj.loc[:, 'accer'] = leader_cumu_dis['accer'].to_numpy()
    out_leader_trj.loc[:, 'remove_outlier_accer'] = leader_cumu_dis['remove_outlier_accer'].to_numpy()
    out_leader_trj.loc[:, 'filter_accer'] = leader_cumu_dis['filter_accer'].to_numpy()
    out_leader_trj.loc[:, 'wavelet_filter_accer'] = leader_cumu_dis['wavelet_filter_accer'].to_numpy()
    out_leader_trj.loc[:, 'jerk'] = leader_cumu_dis['jerk'].to_numpy()
    out_leader_trj.loc[:, 'filter_jerk'] = leader_cumu_dis['filter_jerk'].to_numpy()
    out_leader_trj.loc[:, 'wavelet_filter_jerk'] = leader_cumu_dis['wavelet_filter_jerk'].to_numpy()
    # speed based
    out_follower_trj.loc[:, 'speed_based_position'] = follower_cumu_dis['speed_based_cumu_dis'].to_numpy()
    out_follower_trj.loc[:, 'speed_based_speed'] = follower_cumu_dis['speed_based_speed'].to_numpy()
    out_follower_trj.loc[:, 'speed_based_accer'] = follower_cumu_dis['speed_based_accer'].to_numpy()
    out_follower_trj.loc[:, 'speed_based_jerk'] = follower_cumu_dis['speed_based_jerk'].to_numpy()
    out_leader_trj.loc[:, 'speed_based_position'] = leader_cumu_dis['speed_based_cumu_dis'].to_numpy() + initial_dis
    out_leader_trj.loc[:, 'speed_based_speed'] = leader_cumu_dis['speed_based_speed'].to_numpy()
    out_leader_trj.loc[:, 'speed_based_accer'] = leader_cumu_dis['speed_based_accer'].to_numpy()
    out_leader_trj.loc[:, 'speed_based_jerk'] = leader_cumu_dis['speed_based_jerk'].to_numpy()
    # plot speed and acc figure
    before_and_after_remove_outlier_plot(out_follower_trj)
    before_and_after_remove_outlier_plot(out_leader_trj)
    before_and_after_filtering_plot(out_follower_trj)
    before_and_after_filtering_plot(out_leader_trj)
    # save cf paired trj
    # all_seg_paired_cf_trj = pd.concat([all_seg_paired_cf_trj, pd.concat([out_leader_trj, out_follower_trj])])
    all_seg_paired_cf_trj_with_comparison = all_seg_paired_cf_trj_with_comparison.append(
        pd.concat([out_leader_trj, out_follower_trj]))
    out_follower_trj_final = out_follower_trj.loc[:,
                             ['segment_id', 'local_veh_id', 'length', 'local_time', 'follower_id', 'leader_id',
                              'filter_pos', 'filter_speed', 'filter_accer']]
    out_follower_trj_final.columns = ['segment_id', 'local_veh_id', 'length', 'local_time', 'follower_id', 'leader_id',
                                      'filter_pos', 'filter_speed', 'filter_accer']
    out_leader_trj_final = out_leader_trj.loc[:,
                           ['segment_id', 'local_veh_id', 'length', 'local_time', 'follower_id', 'leader_id',
                            'filter_pos', 'filter_speed', 'filter_accer']]
    out_leader_trj_final.columns = ['segment_id', 'local_veh_id', 'length', 'local_time', 'follower_id', 'leader_id',
                                    'filter_pos', 'filter_speed', 'filter_accer']
    all_seg_paired_cf_trj_final = all_seg_paired_cf_trj_final.append(
        pd.concat([out_leader_trj_final, out_follower_trj_final]))
    # plot the car following trj of both follower and leader
    cf_paired_trj_plot(out_leader_trj_final, out_follower_trj_final, av_label)


def cf_pair_coord_trans(seg_trj_in, follower_id_in, leader_id_in, av_related_label):
    # extract all cf pairs in one segment
    # the input seg_trj_in is already with local id
    # av_related_label determines if there is av involed
    # return the paired trj with transformed coordination with format of ['segment_id', 'local_veh_id', 'length','local_time','follower_id', 'leader_id', 'position', 'speed', 'accer']
    follower_trj = seg_trj_in[seg_trj_in['veh_id'] == follower_id_in]
    leader_trj = seg_trj_in[seg_trj_in['veh_id'] == leader_id_in]
    ego_trj = seg_trj_in[seg_trj_in['veh_id'] == 0]
    if av_related_label:
        # process av related pair
        if follower_id_in == 0 and leader_id_in == 0:
            # this segment is not suitable for cf (av related)
            pass
        elif follower_id_in == 0 and leader_id_in != 0:
            # AV-HV pair
            pair_cf_coord_cal(leader_id_in, leader_trj, 0, ego_trj, 0)
        elif follower_id_in != 0 and leader_id_in == 0:
            # HV-AV pair
            pair_cf_coord_cal(0, ego_trj, follower_id_in, follower_trj, 1)
        else:
            # both AV-HV pair and HV-AV pair
            pair_cf_coord_cal(leader_id_in, leader_trj, 0, ego_trj, 0)
            pair_cf_coord_cal(0, ego_trj, follower_id_in, follower_trj, 1)
    else:
        # process HV-HV pair
        pair_cf_coord_cal(leader_id_in, leader_trj, follower_id_in, follower_trj, 2)


def cf_paired_trj_plot(leader_trj_in, follower_trj_in, av_label):
    # av_label is to determine whether av is leader or follower (0 for follower, 1 for leader, 2 for non-av)
    # the format of the trajectory is pandas dataframe
    # for av_label: 0 means AV-HV, 1 means HV-AV, 2 means HV-HV
    current_segment_id = int(leader_trj_in['segment_id'].iloc[0])
    current_leader_id = int(leader_trj_in['local_veh_id'].iloc[0])
    current_follower_id = int(follower_trj_in['local_veh_id'].iloc[0])
    if av_label == 0:
        follower_line = '-r'
        leader_line = '--b'
        follower_label = 'AV Follower'
        leader_label = 'HV Leader'
        trj_title = 'AV' + '-HV' + str(current_leader_id)
        trj_save_title = 'figure_save/trajectory_process/position_time_plot/av_hv/' + 'Segment_' + str(
            current_segment_id) + '_' + trj_title + '_position_time_plot.png'
    elif av_label == 1:
        follower_line = '-b'
        leader_line = '--r'
        follower_label = 'HV Follower'
        leader_label = 'AV Leader'
        trj_title = 'HV' + str(current_follower_id) + '-AV'
        trj_save_title = 'figure_save/trajectory_process/position_time_plot/hv_av/' + 'Segment_' + str(
            current_segment_id) + '_' + trj_title + '_position_time_plot.png'
    else:
        follower_line = '-b'
        leader_line = '--b'
        follower_label = 'HV Follower'
        leader_label = 'HV Leader'
        trj_title = 'HV' + str(current_follower_id) + '-HV' + str(current_leader_id)
        trj_save_title = 'figure_save/trajectory_process/position_time_plot/hv_hv/' + 'Segment_' + str(
            current_segment_id) + '_' + trj_title + '_position_time_plot.png'
    plt.subplot(3, 1, 1)
    plt.plot(follower_trj_in['local_time'], follower_trj_in['filter_pos'], follower_line, linewidth=0.5, label=follower_label)
    plt.plot(leader_trj_in['local_time'], leader_trj_in['filter_pos'], leader_line, linewidth=0.5, label=leader_label)
    plt.ylabel('Position (m)')
    plt.legend(prop={'size': 6})
    trj_title = 'Segment ' + str(current_segment_id) + ' ' + trj_title + ' Trajectory'
    plt.title(trj_title)
    plt.subplot(3, 1, 2)
    plt.plot(follower_trj_in['local_time'], follower_trj_in['filter_speed'], follower_line, linewidth=0.5, label=follower_label)
    plt.plot(leader_trj_in['local_time'], leader_trj_in['filter_speed'], leader_line, linewidth=0.5, label=leader_label)
    plt.ylabel('Speed (m/s)')
    plt.legend(prop={'size': 6})
    plt.ylim([0, 35])
    plt.subplot(3, 1, 3)
    plt.plot(follower_trj_in['local_time'], follower_trj_in['filter_accer'], follower_line, linewidth=0.5, label=follower_label)
    plt.plot(leader_trj_in['local_time'], leader_trj_in['filter_accer'], leader_line, linewidth=0.5, label=leader_label)
    plt.legend(prop={'size': 6})
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s2)')
    plt.ylim([-5, 5])
    plt.savefig(trj_save_title, dpi=600)
    plt.close('all')


def cf_trj_info_cal(all_seg_paired_cf_trj_in):
    # calculate car following measurements
    # output format
    # ['segment_id', 'local_veh_id', 'length','local_time','follower_id', 'leader_id', 'position', 'speed', 'accer',
    #  'cf_label', 'space_hwy', 'net_distance', 'time_hwy', 'speed_diff', 'TTC', 'DRAC']
    # cf_label: 0 for AV-HV, 1 for HV-AV
    # net_distance = space_hwy - 0.5*follower_length - 0.5*leader_length
    # time_hwy = space_hwy/follower_speed
    # speed_diff = follower_speed - leader_speed
    # Time To Collision: TTC = speed_diff/net_distance
    # Deceleration Required to Avoid Crash: DRAC = (speed_diff ** 2) / net_distance
    def single_cf_pair_info_cal(follower_trj_in, leader_trj_in, cf_label_in):
        global all_seg_cf_info
        # input format ['segment_id', 'local_veh_id', 'length','local_time','follower_id', 'leader_id', 'position', 'speed', 'accer']
        out_cf_info = follower_trj_in.copy(deep=True)
        out_cf_info['cf_label'] = cf_label_in
        out_cf_info['space_hwy'] = 0
        out_cf_info['net_distance'] = 0
        out_cf_info['time_hwy'] = 0
        out_cf_info['speed_diff'] = 0
        out_cf_info['TTC'] = 0
        out_cf_info['DRAC'] = 0
        for i in range(len(out_cf_info['segment_id'])):
            current_time = out_cf_info['local_time'].iloc[i]
            l_time_match = abs(leader_trj_in['local_time'] - current_time) <= 0.001
            matched_leader_trj = leader_trj_in.loc[l_time_match, :]
            if len(matched_leader_trj) > 0:
                space_hwy = matched_leader_trj['filter_pos'].iloc[0] - out_cf_info['filter_pos'].iloc[i]
                out_cf_info['space_hwy'].iloc[i] = space_hwy
                net_distance = space_hwy - 0.5 * matched_leader_trj['length'].iloc[0] - 0.5 * \
                               out_cf_info['length'].iloc[i]
                out_cf_info['net_distance'].iloc[i] = net_distance
                if out_cf_info['filter_speed'].iloc[i] <= 0.1:
                    out_cf_info['time_hwy'].iloc[i] = 1000000
                else:
                    out_cf_info['time_hwy'].iloc[i] = space_hwy / out_cf_info['filter_speed'].iloc[i]
                speed_diff = out_cf_info['filter_speed'].iloc[i] - matched_leader_trj['filter_speed'].iloc[0]
                out_cf_info['speed_diff'].iloc[i] = speed_diff
                if speed_diff < 0:
                    out_cf_info['TTC'].iloc[i] = 0
                    out_cf_info['DRAC'].iloc[i] = 0
                else:
                    out_cf_info['TTC'].iloc[i] = net_distance / speed_diff
                    out_cf_info['DRAC'].iloc[i] = (speed_diff ** 2) / net_distance
        all_seg_cf_info = all_seg_cf_info.append(out_cf_info)

    # ----- av-related cf info -----
    all_seg_av_hv_trj = all_seg_paired_cf_trj_in.loc[all_seg_paired_cf_trj_in['follower_id'] == 0, :]
    all_seg_hv_av_trj = all_seg_paired_cf_trj_in.loc[all_seg_paired_cf_trj_in['leader_id'] == 0, :]
    av_hv_seg_id = pd.unique(all_seg_av_hv_trj['segment_id'])
    for id1 in av_hv_seg_id:
        segment_print = 'Now in AV-HV segment: ' + str(id1)
        print(segment_print)
        current_seg_trj = all_seg_av_hv_trj.loc[all_seg_av_hv_trj['segment_id'] == id1, :]
        follower_id = current_seg_trj['follower_id'].iloc[0]
        leader_id = current_seg_trj['leader_id'].iloc[0]
        flollower_trj = current_seg_trj.loc[current_seg_trj['local_veh_id'] == follower_id, :]
        leader_trj = current_seg_trj.loc[current_seg_trj['local_veh_id'] == leader_id, :]
        single_cf_pair_info_cal(flollower_trj, leader_trj, 0)
    follower_av_seg_id = pd.unique(all_seg_hv_av_trj['segment_id'])
    for id1 in follower_av_seg_id:
        segment_print = 'Now in HV-AV segment: ' + str(id1)
        print(segment_print)
        current_seg_trj = all_seg_hv_av_trj.loc[all_seg_hv_av_trj['segment_id'] == id1, :]
        follower_id = current_seg_trj['follower_id'].iloc[0]
        leader_id = current_seg_trj['leader_id'].iloc[0]
        flollower_trj = current_seg_trj.loc[current_seg_trj['local_veh_id'] == follower_id, :]
        leader_trj = current_seg_trj.loc[current_seg_trj['local_veh_id'] == leader_id, :]
        single_cf_pair_info_cal(flollower_trj, leader_trj, 1)
    # ----- hv-hv cf info -----
    l1 = all_seg_paired_cf_trj_in['follower_id'] != 0
    l2 = all_seg_paired_cf_trj_in['leader_id'] != 0
    all_seg_hv_hv_leader_trj = all_seg_paired_cf_trj_in.loc[l1 & l2, :]
    hv_hv_seg_id = pd.unique(all_seg_hv_hv_leader_trj['segment_id'])
    for id1 in hv_hv_seg_id:
        segment_print = 'Now in HV-HV segment: ' + str(id1)
        print(segment_print)
        current_seg_trj = all_seg_hv_hv_leader_trj.loc[all_seg_hv_hv_leader_trj['segment_id'] == id1, :]
        all_follower_id = pd.unique(current_seg_trj['follower_id'])
        for id2 in all_follower_id:
            # note that one segment may have multiple hv-hv pairs
            current_pair_trj = current_seg_trj.loc[current_seg_trj['follower_id'] == id2, :]
            follower_id = current_pair_trj['follower_id'].iloc[0]
            leader_id = current_pair_trj['leader_id'].iloc[0]
            flollower_trj = current_pair_trj.loc[current_pair_trj['local_veh_id'] == follower_id, :]
            leader_trj = current_pair_trj.loc[current_pair_trj['local_veh_id'] == leader_id, :]
            single_cf_pair_info_cal(flollower_trj, leader_trj, 2)


def cf_pair_exclude_rules_implementation(all_seg_paired_cf_trj_in, all_seg_cf_info_in):
    # this function verify if a selected CF pair is suitable for car following research
    # this verification is necessary because currently the CF pairs are extracted manually by watching the top view videos and might be error-prone
    # 6 rules are defined in the paper:
    # rule 1: Exclude if there is no leader or follower
    # rule 2: Exclude if the follower or leader is off the Lidar detection range (disappear from the video) for some time
    # rule 3: Exclude if the leader or follower is a bus or heavy truck
    # rule 4: Exclude if the follower changes its leader (either the follower or the leader changes its lane)
    # rule 5: Exclude if follower remains standstill during the entire segment
    # rule 6: Exclude if the car following state is interrupted by turning, parking, stop signs, traffic signals, pedestrians, or other obstacles
    # note that: for rule 1 there is no need to verify because all selected pairs have a follower and a leader
    # note that: for rule 4, since there is no lane mark in the provided dataset, so we are not able to analysis lane changing pairs
    # therefore, only rules 2, 3, 5, 6 are implemented here

    all_seg_paired_cf_trj_verified = all_seg_paired_cf_trj_in.copy(deep=True)
    def single_cf_pair_verification(flollower_trj_in, leader_trj_in, follower_cf_info_in):
        # this function implement rules 2, 3, 5, 6
        # output is 0 or 1: 0 denotes this pair is valid, 1 denotes this pair will be removed
        output_label = 0  # default value is 0
        flollower_trj_in.reset_index(inplace=True)
        leader_trj_in.reset_index(inplace=True)
        follower_cf_info_in.reset_index(inplace=True)
        # rule 2
        for i in range(1, len(flollower_trj_in.loc[:, 'segment_id'])):
            # if the time difference between two consecutive points is larger than 0.2s, then this pair is excluded
            if flollower_trj_in.loc[i, 'local_time'] - flollower_trj_in.loc[i - 1, 'local_time'] >= 0.2:
                output_label = 1
                print('Rule 2 excluded')
                return output_label
        for j in range(1, len(leader_trj_in.loc[:, 'segment_id'])):
            # if the time difference between two consecutive points is larger than 0.2s, then this pair is excluded
            if leader_trj_in.loc[j, 'local_time'] - leader_trj_in.loc[j - 1, 'local_time'] >= 0.2:
                output_label = 1
                print('Rule 2 excluded')
                return output_label
        # rule 3
        large_vehicle_length_threshold = 8
        if flollower_trj_in.loc[0, 'length'] >= large_vehicle_length_threshold:
            output_label = 1
            print('Rule 3 excluded')
            return output_label
        if leader_trj_in.loc[0, 'length'] >= large_vehicle_length_threshold:
            output_label = 1
            print('Rule 3 excluded')
            return output_label
        # rule 5
        if flollower_trj_in.loc[:, 'filter_speed'].max() <= 0.1:
            # the case where the follower is always standstill
            output_label = 1
            print('Rule 5 excluded')
            return output_label
        # rule 6
        # based on the slope of v-s curve, if the slope is obviously negative, then this pair is excluded
        detection_length = 50  # calculate the slope every 50 points
        slope_threhold = -0.5  # if the slope of v-s curve is smaller than this threshold, then this pair is excluded
        for i in range(len(follower_cf_info_in.loc[:, 'segment_id']) - detection_length):
            # monotonic test, only in the case of monotonic v-s curve will be slope be calculated
            l_speed = follower_cf_info_in.loc[i:i+detection_length - 1, 'filter_speed'].is_monotonic_increasing or \
                      follower_cf_info_in.loc[i:i+detection_length - 1, 'filter_speed'].is_monotonic_decreasing
            l_spacing = follower_cf_info_in.loc[i:i+detection_length - 1, 'space_hwy'].is_monotonic_increasing or \
                      follower_cf_info_in.loc[i:i+detection_length - 1, 'space_hwy'].is_monotonic_decreasing
            if l_speed and l_spacing:
                v_data = follower_cf_info_in.loc[i:i+detection_length - 1, 'filter_speed'].values.reshape(-1, 1)
                s_data = follower_cf_info_in.loc[i:i+detection_length - 1, 'space_hwy'].values.reshape(-1, 1)
                current_regression = LinearRegression()
                current_regression.fit(s_data, v_data)
                current_slope = current_regression.coef_[0]
                if current_slope <= slope_threhold:
                    output_label = 1
                    print('Rule 6 excluded')
                    return output_label
        return output_label
    all_seg_id = pd.unique(all_seg_paired_cf_trj_in['segment_id'])
    for id1 in all_seg_id:
        current_seg_trj = all_seg_paired_cf_trj_in.loc[all_seg_paired_cf_trj_in['segment_id'] == id1, :]
        current_seg_cf_info = all_seg_cf_info_in.loc[all_seg_cf_info_in['segment_id'] == id1, :]
        all_follower_id = pd.unique(current_seg_trj['follower_id'])
        for id2 in all_follower_id:
            current_pair_trj = current_seg_trj.loc[current_seg_trj['follower_id'] == id2, :]
            current_follower_cf_info = current_seg_cf_info.loc[current_seg_cf_info['follower_id'] == id2, :]
            follower_id = current_pair_trj['follower_id'].iloc[0]
            leader_id = current_pair_trj['leader_id'].iloc[0]
            segment_print = 'Now in segment: ' + str(id1) + ' Follower:' + str(follower_id) + ' Leader:' + str(leader_id)
            print(segment_print)
            flollower_trj = current_pair_trj.loc[current_pair_trj['local_veh_id'] == follower_id, :]
            leader_trj = current_pair_trj.loc[current_pair_trj['local_veh_id'] == leader_id, :]
            verification_result = single_cf_pair_verification(flollower_trj, leader_trj, current_follower_cf_info)
            if verification_result:
                # remove this pair
                l_segment_id = all_seg_paired_cf_trj_verified['segment_id'] == id1
                l_follower_id = all_seg_paired_cf_trj_verified['follower_id'] == follower_id
                l_leader_id = all_seg_paired_cf_trj_verified['leader_id'] == leader_id
                l_overall = (l_segment_id & l_follower_id) & l_leader_id
                all_seg_paired_cf_trj_verified.drop(all_seg_paired_cf_trj_verified[l_overall].index, inplace=True)
    all_seg_paired_cf_trj_verified.to_csv('data_save/all_seg_paired_cf_trj_verified.csv')


def lidar_detection_range():
    # calculate the maximum detection range in each segment for each type of objects
    all_segment_lidar_original_lidar_distance = pd.DataFrame()
    all_segment_lidar_original_lidar_distance_dict = []
    all_segment_lidar_maximum_detection_range = pd.DataFrame()
    temp_save_step = 100
    temp_save_segment_id_start = 0
    for file_i in range(1, 11):
        obj_info_read_name = 'data_save/' + 'segment_' + str(temp_save_segment_id_start) + '_' + str(
            temp_save_segment_id_start + temp_save_step) + '_all_frame_all_object_info.csv'
        temp_obj_info = pd.read_csv(obj_info_read_name, header=0)
        all_seg_id = pd.unique(temp_obj_info['segment_id'])
        for seg_id in all_seg_id:
            segment_id_print = 'Now in segment: ' + str(seg_id)
            print(segment_id_print)
            l_seg_id = temp_obj_info['segment_id'] == seg_id
            seg_obj_info = temp_obj_info[l_seg_id]
            one_seg_lidar_range = pd.DataFrame()  # calculate the maximum lidar range in all frames
            one_seg_lidar_range_dict = []  # calculate the maximum lidar range in all frames
            seg_num_frame = seg_obj_info.loc[:, 'frame_label'].max()
            for frame_i in range(1, seg_num_frame+1):
                l_frame = seg_obj_info.loc[:, 'frame_label'] == frame_i
                frame_obj_info = seg_obj_info.loc[l_frame, :]
                frame_all_obj_id = pd.unique(frame_obj_info.loc[:, 'obj_id'])
                ego_x0 = frame_obj_info.loc[frame_obj_info['obj_id'] == 'ego', 'global_center_x'].iat[0]
                ego_y0 = frame_obj_info.loc[frame_obj_info['obj_id'] == 'ego', 'global_center_y'].iat[0]
                ego_z0 = frame_obj_info.loc[frame_obj_info['obj_id'] == 'ego', 'global_center_z'].iat[0]
                for obj_id in frame_all_obj_id:
                    if obj_id == 'ego':
                        continue
                    else:
                        # single_obj_dis = pd.DataFrame(np.zeros((1, 5)), columns=['segment_id', 'frame_id', 'obj_id', 'obj_type', 'obj_distance'])
                        single_obj_dis_dict = {}
                        obj_x = frame_obj_info.loc[frame_obj_info['obj_id'] == obj_id, 'global_center_x'].iat[0]
                        obj_y = frame_obj_info.loc[frame_obj_info['obj_id'] == obj_id, 'global_center_y'].iat[0]
                        obj_z = frame_obj_info.loc[frame_obj_info['obj_id'] == obj_id, 'global_center_z'].iat[0]
                        single_obj_dis_dict['segment_id'] = seg_id
                        single_obj_dis_dict['frame_id'] = frame_i
                        single_obj_dis_dict['obj_id'] = obj_id
                        single_obj_dis_dict['obj_type'] = frame_obj_info.loc[frame_obj_info['obj_id'] == obj_id, 'obj_type'].iat[0]
                        single_obj_dis_dict['obj_distance'] = math.sqrt((obj_x - ego_x0)**2 + (obj_y - ego_y0)**2 + (obj_z - ego_z0)**2)
                        one_seg_lidar_range_dict.append(single_obj_dis_dict)
            one_seg_lidar_range = pd.DataFrame(one_seg_lidar_range_dict)
            # all_segment_lidar_original_lidar_distance = all_segment_lidar_original_lidar_distance.append(one_seg_lidar_range)
            all_segment_lidar_original_lidar_distance_dict.append(one_seg_lidar_range_dict)
            if len(one_seg_lidar_range) >0:
                # all_segment_lidar_original_lidar_distance.to_csv(('data_save/all_segment_lidar_original_lidar_distance.csv'))
                one_seg_max_dis = pd.DataFrame(np.zeros((1, 4)), columns=['segment_id', 'vehicle_max', 'bicycle_max', 'pedestrian_max'])
                one_seg_max_dis.loc[0, 'segment_id'] = seg_id
                seg_vehicle_lidar_range = one_seg_lidar_range.loc[one_seg_lidar_range['obj_type'] == 'vehicle', :]
                if len(seg_vehicle_lidar_range) > 0:
                    one_seg_max_dis.loc[0, 'vehicle_max'] = seg_vehicle_lidar_range.loc[:, 'obj_distance'].max()
                seg_bicycle_lidar_range = one_seg_lidar_range.loc[one_seg_lidar_range['obj_type'] == 'bicycle', :]
                if len(seg_bicycle_lidar_range) > 0:
                    one_seg_max_dis.loc[0, 'bicycle_max'] = seg_bicycle_lidar_range.loc[:, 'obj_distance'].max()
                seg_pedestrian_lidar_range = one_seg_lidar_range.loc[one_seg_lidar_range['obj_type'] == 'pedestrian', :]
                if len(seg_pedestrian_lidar_range) > 0:
                    one_seg_max_dis.loc[0, 'pedestrian_max'] = seg_pedestrian_lidar_range.loc[:, 'obj_distance'].max()
                all_segment_lidar_maximum_detection_range = all_segment_lidar_maximum_detection_range.append(one_seg_max_dis)
            # all_segment_lidar_maximum_detection_range.to_csv(('data_save/all_segment_lidar_maximum_detection_range.csv'))
        temp_save_segment_id_start += temp_save_step
        all_segment_lidar_original_lidar_distance = pd.DataFrame(all_segment_lidar_original_lidar_distance_dict)
        temp_save_name = 'data_save/all_segment_lidar_original_lidar_distance.csv'
        all_segment_lidar_original_lidar_distance.to_csv(temp_save_name)
        temp_save_name = 'data_save/all_segment_lidar_maximum_detection_range.csv'
        all_segment_lidar_maximum_detection_range.to_csv(temp_save_name)


def lidar_detection_range_aggregated_results():
    seg_max_dis = pd.read_csv('data_save/all_segment_lidar_maximum_detection_range.csv', header=0)
    all_segment_aggregated_lidar_range = pd.DataFrame(np.zeros((4, 3)), columns=['vehicle_max', 'bicycle_max', 'pedestrian_max']
                                                      , index=['max', 'min', 'median', 'mad'])
    l_vehicle_max = seg_max_dis.loc[:, 'vehicle_max'] > 0.01
    all_segment_aggregated_lidar_range.loc['max', 'vehicle_max'] = seg_max_dis.loc[l_vehicle_max, 'vehicle_max'].max()
    all_segment_aggregated_lidar_range.loc['min', 'vehicle_max'] = seg_max_dis.loc[l_vehicle_max, 'vehicle_max'].min()
    all_segment_aggregated_lidar_range.loc['median', 'vehicle_max'] = seg_max_dis.loc[l_vehicle_max, 'vehicle_max'].median()
    all_segment_aggregated_lidar_range.loc['mad', 'vehicle_max'] = seg_max_dis.loc[l_vehicle_max, 'vehicle_max'].mad()
    l_bicycle_max = seg_max_dis.loc[:, 'bicycle_max'] > 0.01
    all_segment_aggregated_lidar_range.loc['max', 'bicycle_max'] = seg_max_dis.loc[l_bicycle_max, 'bicycle_max'].max()
    all_segment_aggregated_lidar_range.loc['min', 'bicycle_max'] = seg_max_dis.loc[l_bicycle_max, 'bicycle_max'].min()
    all_segment_aggregated_lidar_range.loc['median', 'bicycle_max'] = seg_max_dis.loc[l_bicycle_max, 'bicycle_max'].median()
    all_segment_aggregated_lidar_range.loc['mad', 'bicycle_max'] = seg_max_dis.loc[l_bicycle_max, 'bicycle_max'].mad()
    l_pedestrian_max = seg_max_dis.loc[:, 'pedestrian_max'] > 0.01
    all_segment_aggregated_lidar_range.loc['max', 'pedestrian_max'] = seg_max_dis.loc[
        l_pedestrian_max, 'pedestrian_max'].max()
    all_segment_aggregated_lidar_range.loc['min', 'pedestrian_max'] = seg_max_dis.loc[
        l_pedestrian_max, 'pedestrian_max'].min()
    all_segment_aggregated_lidar_range.loc['median', 'pedestrian_max'] = seg_max_dis.loc[
        l_pedestrian_max, 'pedestrian_max'].median()
    all_segment_aggregated_lidar_range.loc['mad', 'pedestrian_max'] = seg_max_dis.loc[
        l_pedestrian_max, 'pedestrian_max'].mad()
    temp_save_name = 'data_save/all_segment_aggregated_lidar_range.csv'
    all_segment_aggregated_lidar_range.to_csv(temp_save_name)


if __name__ == '__main__':

    # All required input files has been shared in https://data.mendeley.com/datasets/wfn2c3437n/1
    # Some directories need to be created before running: 2 directories "figure_save" and "data_save" need to be created
    # in the  directory where the code is located;
    # under "figure_save", 5 directories need to be created: "temp_cam_pic", "cam_video", "temp_top_view_figure","top_view_video",
    # "trajectory_process"
    # under "trajectory_process", 4 directories need to created: "outlier_pattern_and_removing", "before_and_after_remove_outlier_plot",
    # "before_and_after_filtering_plot", "position_time_plot"
    # under "position_time_plot", 3 directories need to be created: "av_hv", "hv_av", "hv_hv"
    # Note that each part of the code can be run seperately, and each part has a label variable to control the run
    # By default all the labels are set to 0, to switch on one part the user needs to set the corresponding label to 1

    # ******************** Stage 1: collect lidar object information from original dataset ********************

    # to determine if camera video should be made in this run
    camera_video_generation_label = 0
    # to determine if Lidar information should be extracted in this run
    lidar_information_label = 0
    # version 1.2 label
    version_1_2_label = 0  # if processing version 1.2 data, this label should be set to 1
    if lidar_information_label:
        # initialize a list to save the match relationships between segment short local id and gobal unique long name (file name)
        segment_name_list = []
        # initialize a list to store the final processed data
        all_segment_all_frame_all_object_info = []
        if version_1_2_label:
            # this is for waymo data version 1.2 (200 more segments)
            segment_id = 1000  # there are 1000 segments in version 1.1
            all_segment_files = sorted(
                # the file path should be replaced if used by others
                glob.glob('/media/shawn/Prof_Zheng_Group/AV_Data/Waymo/veision1.2/segment*.tfrecord'))
        else:
            all_segment_files = sorted(
                glob.glob('/media/shawn/Prof_Zheng_Group/AV_Data/Waymo/waymo_open_data/segment*.tfrecord'))
            segment_id = 0
        for segment_file in all_segment_files:
            segment_id += 1
            segment_name = os.path.basename(segment_file)
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
            for frame_data in segment_dataset:
                frame_label += 1
                print_name = 'Now in frame: ' + str(frame_label)
                print(print_name)
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(frame_data.numpy()))
                # ---------- camera image generation ----------
                if camera_video_generation_label:
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
            # ---------- camera video generation ----------
            if camera_video_generation_label:
                camera_video_generation()
            segment_name_list.append(single_segment_name_dict)
            # save up-to-date results every 100 segments to avoid too big single file
            save_step = 100
            if segment_id % save_step == 0:
                segment_name_pd = pd.DataFrame(segment_name_list)
                all_segment_all_frame_all_object_info_pd = pd.DataFrame(all_segment_all_frame_all_object_info)
                final_trj_result_format()
                if version_1_2_label:
                    segment_name_pd.to_csv('data_save/segment_global_name_and_local_id_match_version_1_2.csv',
                                           index=False)
                    save_name = 'data_save/segment_' + str(segment_id - save_step) + '_' + str(
                        segment_id) + '_all_frame_all_object_info_version_1_2.csv'
                    all_segment_all_frame_all_object_info_pd.to_csv(save_name, index=False)
                else:
                    segment_name_pd.to_csv('data_save/segment_global_name_and_local_id_match.csv', index=False)
                    save_name = 'data_save/segment_' + str(segment_id - save_step) + '_' + str(
                        segment_id) + '_all_frame_all_object_info.csv'
                    all_segment_all_frame_all_object_info_pd.to_csv(save_name, index=False)
                # must restore contanier's state, or the memory could spill
                all_segment_all_frame_all_object_info = []

    # ******************** Stage 2: visualization of lidar information ********************

    # ---------- process calulated trajectories from the csv lidar information file ----------
    local_veh_id_generation_label = 0  # a local vehicle ID will make each object more identifiable
    if local_veh_id_generation_label:
        # ---------- process calulated trajectories from the csv file ----------
        all_segment_all_vehicle_object_id_and_local_id_match = []
        save_step = 100
        if version_1_2_label:
            save_segment_id_start = 1000
            total_steps = 2
        else:
            save_segment_id_start = 0
            total_steps = 11
        for i in range(1, total_steps):
            read_name = 'data_save/' + 'segment_' + str(save_segment_id_start) + '_' + str(
                save_segment_id_start + save_step) + '_all_frame_all_object_info.csv'
            temp_trj = pd.read_csv(read_name, header=0)
            all_segment_id = pd.unique(temp_trj['segment_id'])
            for single_seg_id in all_segment_id:
                segment_print = 'Now in segment (generating local id): ' + str(single_seg_id)
                print(segment_print)
                l_seg_id = temp_trj['segment_id'] == single_seg_id
                seg_trj = temp_trj[l_seg_id]
                all_segment_veh_id = pd.unique(seg_trj['obj_id'])
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
        if version_1_2_label:
            all_segment_all_vehicle_object_id_and_local_id_match_pd.to_csv(
                'data_save/all_segment_all_vehicle_object_id_and_local_id_match_version_1_2.csv', index=False)
        else:
            all_segment_all_vehicle_object_id_and_local_id_match_pd.to_csv(
                'data_save/all_segment_all_vehicle_object_id_and_local_id_match.csv', index=False)
    # # to determine if the top view video should be generated in this run
    top_view_video_generation_label = 0
    if top_view_video_generation_label:
        save_step = 100
        if version_1_2_label:
            save_segment_id_start = 1000
            total_steps = 2
        else:
            save_segment_id_start = 0
            total_steps = 11
        for i in range(1, total_steps):
            trj_read_name = 'data_save/' + 'segment_' + str(save_segment_id_start) + '_' + str(
                save_segment_id_start + save_step) + '_all_frame_all_object_info.csv'
            temp_trj = pd.read_csv(trj_read_name, header=0,
                                   dtype={'global_center_x': np.float64, 'global_center_y': np.float64})
            # this match file has three columns [segment_id, obj_id, local_id]
            if version_1_2_label:
                veh_name_id_match = pd.read_csv(
                    'data_save/all_segment_all_vehicle_object_id_and_local_id_match_version_1_2.csv', header=0)
            else:
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

    # ******************** Stage 3: choose car following pairs based on videos ********************
    # this part is manually done by carefully checking the top view videos
    # the results are save in two ".csv" files
    # "AV_HV_and_HV_AV_CF_pair_record_with_large_vehicle.csv"
    # "HV_HV_pair_record_with_large_vehicle.csv"

    # ******************** Stage 4: extract the trajectories of the selected CF pairs, process and enhance the data quality ********************
    cf_analysis_trj_gengeration_label = 0
    cf_trj_info_generation_label = 0
    # import global object id and local id match table
    temp_id_match_path = 'data_save/all_segment_all_vehicle_object_id_and_local_id_match.csv'
    # format of this match table is [segment_id, veh_name, local_id]
    veh_name_global_local_match = pd.read_csv(temp_id_match_path, header=0)
    # ---------- CF analysis trajectory (CF pair extraction and coordinate transformation) ----------
    if cf_analysis_trj_gengeration_label:
        all_seg_paired_cf_trj_with_comparison = pd.DataFrame()
        all_seg_paired_cf_trj_final = pd.DataFrame()
        all_outlier_record = pd.DataFrame()
        save_step = 100
        save_segment_id_start = 0
        error_start_segment_id = 0
        for i in range(1, 11):
            trj_read_name = 'data_save/' + 'segment_' + str(save_segment_id_start) + '_' + str(
                save_segment_id_start + save_step) + '_all_frame_all_object_info.csv'
            temp_trj = pd.read_csv(trj_read_name, header=0,
                                   dtype={'global_center_x': np.float64, 'global_center_y': np.float64})
            # format of input trajectory:
            # [segment_id, frame_label, time_of_day, location, weather, laser_veh_count, obj_type, obj_id, global_time_stamp, local_time_stamp,
            #  local_center_x, local_center_y, local_center_z, global_center_x, global_center_y, global_center_z, length, width,
            #  height, heading, speed_x, speed_y, accel_x, accel_y, angular_speed]
            # this match file has three columns [segment_id, obj_id, local_id]
            veh_name_id_match = pd.read_csv('data_save/all_segment_all_vehicle_object_id_and_local_id_match.csv',
                                            header=0)
            veh_name_id_match_dict = veh_name_id_match.set_index('obj_id').to_dict()['local_id']
            veh_name_id_match_dict['ego'] = 0
            # convert veh name to local id
            temp_trj = temp_trj.loc[temp_trj['obj_type'] == 'vehicle', :]
            new_veh_id = temp_trj['obj_id'].map(veh_name_id_match_dict)
            temp_trj['veh_id'] = new_veh_id
            # ----- extract AV related CF paired trajectory -----
            # this CF label file has 5 columns [segment_id, follower, leader, follower_code, leader_code], code means the exclusion reason code
            cf_label_match = pd.read_csv('data_save/AV_HV_and_HV_AV_CF_pair_record_with_large_vehicle.csv', header=0)
            all_segment_id = pd.unique(temp_trj['segment_id'])
            for single_seg_id in all_segment_id:
                segment_print = 'Now in segment (AV_related): ' + str(single_seg_id)
                print(segment_print)
                l_seg_id = temp_trj['segment_id'] == single_seg_id
                seg_trj = temp_trj[l_seg_id]
                # get current segment's label
                seg_label = cf_label_match[cf_label_match['segment_id'] == single_seg_id]
                if single_seg_id >= error_start_segment_id:
                    cf_pair_coord_trans(seg_trj, seg_label['follower'].iloc[0], seg_label['leader'].iloc[0], 1)
            # ----- extract HV-HV related CF paired trajectory -----
            cf_label_match = pd.read_csv('data_save/HV_HV_pair_record_with_large_vehicle.csv', header=0)
            all_segment_id = pd.unique(temp_trj['segment_id'])
            for single_seg_id in all_segment_id:
                segment_print = 'Now in segment (HV-HV): ' + str(single_seg_id)
                print(segment_print)
                l_seg_id = temp_trj['segment_id'] == single_seg_id
                seg_trj = temp_trj[l_seg_id]
                # get current segment's label
                seg_label = cf_label_match[cf_label_match['segment_id'] == single_seg_id]
                if len(seg_label) > 0:
                    for j in range(len(seg_label['segment_id'])):
                        if single_seg_id >= error_start_segment_id:
                            cf_pair_coord_trans(seg_trj, seg_label['follower'].iloc[j], seg_label['leader'].iloc[j], 0)
            # save the output paired cf trj
            all_seg_paired_cf_trj_with_comparison.to_csv('data_save/all_seg_paired_cf_trj_with_comparison_with_large_vehicle.csv',
                                                         index=False)
            all_seg_paired_cf_trj_final.to_csv('data_save/all_seg_paired_cf_trj_final_with_large_vehicle.csv', index=False)
            all_outlier_record.to_csv('data_save/all_outlier_record_with_large_vehicle.csv', index=False)
            save_segment_id_start += save_step
    if cf_trj_info_generation_label:
        all_seg_cf_info = pd.DataFrame()
        all_seg_paired_cf_trj = pd.read_csv('data_save/all_seg_paired_cf_trj_final_with_large_vehicle.csv', header=0)
        cf_trj_info_cal(all_seg_paired_cf_trj)
        all_seg_cf_info.to_csv('data_save/all_seg_cf_info.csv', index=False)
    CF_verification_label = 0
    if CF_verification_label:
        all_seg_cf_info = pd.read_csv('data_save/all_seg_cf_info.csv', header=0)
        all_seg_paired_cf_trj = pd.read_csv('data_save/all_seg_paired_cf_trj_final_with_large_vehicle.csv', header=0)
        cf_pair_exclude_rules_implementation(all_seg_paired_cf_trj, all_seg_cf_info)
    lidar_detection_range_label = 0
    if lidar_detection_range_label:
        lidar_detection_range()
        lidar_detection_range_aggregated_results()
