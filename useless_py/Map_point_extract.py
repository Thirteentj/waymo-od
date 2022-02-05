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


global point_has_been_pointed
point_has_been_pointed = []
def plot_top_view_single_pic_map(trj_in,file_index,scenario_id_in, scenario,target_left,target_right,length,lane_turn_right_id_real=[]):
    global point_has_been_pointed
    #plt.figure(figsize=(10, 7))
    fig, ax = plt.subplots(1,2,figsize=(14,7))
    plt.xlabel('global center x (m)', fontsize=10)
    plt.ylabel('global center y (m)', fontsize=10)
    plt.axis('square')
    plt.xlim([trj_in['center_x'].min() - 1, trj_in['center_x'].max() + 1])
    plt.ylim([trj_in['center_y'].min() - 1, trj_in['center_y'].max() + 1])
    title_name = 'Scenario ' + str(scenario_id_in)
    plt.title(title_name, loc='left')
    plt.xticks(np.arange(round(float(trj_in['center_x'].min())), round(float(trj_in['center_x'].max())), 20),fontsize=5)
    plt.yticks(np.arange(round(float(trj_in['center_y'].min())), round(float(trj_in['center_y'].max())), 20), fontsize=5)
    #ax = plt.subplots(121)
    map_features = scenario.map_features
    road_edge_count = 0
    lane_count = 0
    road_line = 0
    all_element_count = 0
    for single_feature in map_features:
        all_element_count += 1
        id_ = single_feature.id
        #print("id is %d"%id_)
        if list(single_feature.road_edge.polyline)!= []:
            road_edge_count += 1
            single_line_x = []
            single_line_y = []
            # print("road_edge id is %d"%single_feature.id)
            for polyline in single_feature.road_edge.polyline:
                single_line_x.append(polyline.x)
                single_line_y.append(polyline.y)
            ax[0].plot(single_line_x, single_line_y, color='black', linewidth=0.3)  # 道路边界为黑色

        if list(single_feature.lane.polyline)!= []:
            lane_count += 1
            single_line_x = []
            single_line_y = []
            for polyline in single_feature.lane.polyline:
                single_line_x.append(polyline.x)
                single_line_y.append(polyline.y)
            #z1 = np.polyfit(single_line_x,single_line_y,8)
            #p1 = np.poly1d(z1)
            #y_hat = p1(single_line_x)
            #ax.plot(single_line_x,y_hat,color='green', linewidth=0.5)
            if id_ in target_left:
                ax[0].plot(single_line_x, single_line_y, color='green', linewidth=0.5)
                ax[1].plot(single_line_x, single_line_y, color='green', linewidth=0.5)
            elif id_ in target_right:
                if id_ in lane_turn_right_id_real:  #目标交叉口的右转车道
                    ax[0].plot(single_line_x, single_line_y, color='red', linewidth=0.5)
                    ax[1].plot(single_line_x, single_line_y, color='red', linewidth=0.5)
                else:
                    ax[0].plot(single_line_x, single_line_y, color='purple', linewidth=0.5)
                    ax[1].plot(single_line_x, single_line_y, color='purple', linewidth=0.5)
            else:
                ax[0].plot(single_line_x, single_line_y, color='blue', linewidth=0.5)  # 道路中心线为蓝色
            if (single_line_x[0],single_line_y[0]) not in point_has_been_pointed:
                ax[0].text(single_line_x[0], single_line_y[0], id_, fontsize=1.5)
                point_has_been_pointed.append((single_line_x[0],single_line_y[0]))
            else:
                ax[0].text(single_line_x[0]-5, single_line_y[0]-5, id_,color='red', fontsize=1.5)
                point_has_been_pointed.append((single_line_x[0]-5, single_line_y[0]-5))

        if list(single_feature.road_line.polyline)!=[]:
            road_line += 1
            single_line_x = []
            single_line_y = []
            for polyline in single_feature.road_line.polyline:
                single_line_x.append(polyline.x)
                single_line_y.append(polyline.y)
            ax[0].plot(single_line_x, single_line_y, color='black', linestyle='-', linewidth=0.3)  # 道路标线为  虚线

    fig_save_name = 'figure_save/intersection_topo_figure/top_view_segment_'+ str(file_index) + '_scenario_' + str(scenario_id_in) + '_trajectory.jpg'
    plt.savefig(fig_save_name, dpi=600)
    #plt.show()
    plt.close('all')
    return road_edge_count, lane_count, road_line,all_element_count



def get_lane_min_dis(single_scenario_all_feature,map_features_id_list,ego_lane_id,other_lanes,connect_type):
    ego_index = map_features_id_list.index(ego_lane_id)
    ego_lane_info = single_scenario_all_feature[ego_index]
    lane_inter_dis = []   #用于记录本车道最尽头和目标车道最尽头之间的距离，用于判定是否为交叉口内部
    ego_lane_point = ()
    other_lane_point = []
    for other_lane_id in other_lanes:
        other_lane_index = map_features_id_list.index(other_lane_id)
        other_lane_info = single_scenario_all_feature[other_lane_index]
        if connect_type == 'entry':
            x1,y1 = ego_lane_info.lane.polyline[0].x,ego_lane_info.lane.polyline[0].y
            x2,y2 = other_lane_info.lane.polyline[0].x,other_lane_info.lane.polyline[0].y
            ego_lane_point = (ego_lane_info.lane.polyline[0].x,ego_lane_info.lane.polyline[0].y)  #如果是进入的关系，则返回该车道的第一个点
            other_lane_point.append((other_lane_info.lane.polyline[0].x,other_lane_info.lane.polyline[0].y))
        if connect_type == 'exit':
            x1, y1 = ego_lane_info.lane.polyline[-1].x, ego_lane_info.lane.polyline[-1].y
            x2, y2 = other_lane_info.lane.polyline[-1].x, other_lane_info.lane.polyline[-1].y
            ego_lane_point = (
            ego_lane_info.lane.polyline[-1].x, ego_lane_info.lane.polyline[-1].y)  # 如果是驶出的关系，则返回该车道的最后一个点
            other_lane_point.append((other_lane_info.lane.polyline[-1].x, other_lane_info.lane.polyline[-1].y))

        dis = np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
        lane_inter_dis.append(dis)
    return lane_inter_dis,ego_lane_point,other_lane_point

def cal_angle(x1,y1,x2,y2):
    if x2!=x1:
        angle = (math.atan((y2-y1)/(x2-x1)))*180/np.pi
    else:
        angle = 90  #避免斜率不存在的情况
    return angle

def get_lane_angle_chane(polyline_list,ego_lane_id):  #计算
    angle_start = 0
    angle_end = 0
    length = len(polyline_list)
    x_list = []
    y_list = []
    turn_type = 'straight'
    x1, y1 = polyline_list[0].x, polyline_list[0].y
    x4, y4 = polyline_list[-1].x, polyline_list[-1].y
    for polyline in polyline_list:
        x_list.append(polyline.x)
        y_list.append(polyline.y)
    try:
        # print(polyline_list)
        x2, y2 = polyline_list[3].x, polyline_list[3].y
        x3, y3 = polyline_list[-3].x, polyline_list[-3].y
        angle_start = cal_angle(x1,y1, x2, y2)
        angle_end = cal_angle(x3,y3,x4,y4)
        delta_angle = angle_end - angle_start  # 大于0为左转，小于0为右转
    except:
        angle_start = angle_end = delta_angle = None

    #判断左右转信息
    index_mid = int(length/2)
    x_mid = polyline_list[index_mid].x
    y_mid = polyline_list[index_mid].y
    # p1 = np.array((x_mid-x1,y_mid-y1))
    # p2 = np.array((x4-x_mid,y4-y_mid))
    p3 = (x_mid-x1)*(y4-y_mid)-(y_mid-y1)*(x4-x_mid)
    #print(p3)
    if p3 > 0:
        turn_type = 'left'
    elif p3<0:
        turn_type = 'right'
    #print("Turn type is %s"%turn_type)
    return angle_start,angle_end,delta_angle,turn_type

def cal_lane_slpoe(polyline):
    x1,y1 = polyline[0].x,polyline[0].y  #直线起点xy坐标
    x2,y2 = polyline[-1].x,polyline[-1].y  #直线终点xy坐标
    if x2-x1 != 0:
        slope = (y2-y1)/(x2-x1)
    else:
        slope = 90
    return slope

def map_lane_id_all(map_features):
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
    return line_id_ave, line_num, lane_id_ave, lane_num

def map_topo_info_extract(map_features):  #提取车道连接关系等拓扑信息
    single_feature_all_lane_polyline = []  # 一个scenario中所有的车道信息（仅包括散点）
    lane_id_all = []  #一个scenario中所有车道的ID信息
    all_lane_entry_exit_info = []  # 记录所有车道的进出车道信息
    single_map_dict = {}
    lane_turn_left_id = []  #记录交叉口内部所有左转车道ID
    lane_turn_right_id = []  #记录交叉口内部所有右转车道ID

    for single_feature in map_features:  # 先将每根车道线的信息保存成列表
        single_scenario_all_feature.append(single_feature)
        map_features_id_list.append(single_feature.id)

    for single_feature in map_features:
        single_lane_entry_exit_info = {}  # 记录一个车道的进出车道信息，包括与之相邻车道终点与该车道终点的距离（用于和交叉口尺寸阈值进行判定）
        if list(single_feature.lane.polyline) != []:
            ego_lane_id = single_feature.id
            entry_lanes = single_feature.lane.entry_lanes
            exit_lanes = single_feature.lane.exit_lanes
            entry_lanes_dis, ego_lane_point_entry, entry_lane_point = get_lane_min_dis(single_scenario_all_feature,
                                                                                       map_features_id_list,
                                                                                       ego_lane_id, entry_lanes,
                                                                                       'entry')
            exit_lanes_dis, ego_lane_point_exit, exit_lane_point = get_lane_min_dis(single_scenario_all_feature,
                                                                                    map_features_id_list, ego_lane_id,
                                                                                    exit_lanes, 'exit')
            angle_start, angle_end, delta_angle, turn_type = get_lane_angle_chane(single_feature.lane.polyline,
                                                                                  ego_lane_id)  # 该线段的角度变化值
            single_lane_entry_exit_info['file_index'] = file_index
            single_lane_entry_exit_info['scenario_id'] = scenario_label
            single_lane_entry_exit_info['lane_id'] = ego_lane_id
            single_lane_entry_exit_info['angle_start'] = angle_start
            single_lane_entry_exit_info['angle_end'] = angle_end
            single_lane_entry_exit_info['ego_lane_angle_change'] = delta_angle
            single_lane_entry_exit_info['is_a_turn'] = ''
            if delta_angle:
                if abs(delta_angle) > 35:  #由50暂时修正为35
                    # if 120>delta_angle >50: #为左转
                    # if delta_angle > 50:  # 为左转
                    if turn_type == 'left':
                        single_lane_entry_exit_info['is_a_turn'] = 'left'
                        lane_turn_left_id.append(ego_lane_id)
                    # elif -120<delta_angle <-50:
                    # elif delta_angle < -50:
                    elif turn_type == 'right':
                        single_lane_entry_exit_info['is_a_turn'] = 'right'
                        lane_turn_right_id.append(ego_lane_id)  # 整个交叉口右转车道只有四根
                    else:
                        single_lane_entry_exit_info['is_a_turn'] = 'straight'
                else:
                    single_lane_entry_exit_info['is_a_turn'] = 'straight'

            if single_lane_entry_exit_info['is_a_turn'] == 'straight':
                single_lane_entry_exit_info['lane_slope'] = cal_lane_slpoe(single_feature.lane.polyline)  #如果是直线车道，计算这条车道的斜率，用于后续计算交叉口角度
            # single_lane_entry_exit_info['is a turn'] = turn_type
            #相连接车道信息提取
            single_lane_entry_exit_info['ego_lane_point_entry'] = ego_lane_point_entry
            single_lane_entry_exit_info['ego_lane_point_exit'] = ego_lane_point_exit
            single_lane_entry_exit_info['entry_lanes'] = entry_lanes
            single_lane_entry_exit_info['entry_lanes_dis'] = entry_lanes_dis
            single_lane_entry_exit_info['entry_lane_point'] = entry_lane_point
            single_lane_entry_exit_info['exit_lanes'] = exit_lanes
            single_lane_entry_exit_info['exit_lanes_dis'] = exit_lanes_dis
            single_lane_entry_exit_info['exit_lane_point'] = exit_lane_point
            #相邻车道信息提取
            lane_index_all = len(list(single_feature.lane.polyline))  #这条路被分成的小片段序列总数（即散点总数）
            single_lane_entry_exit_info['left_neighbors_id'] = -1  #先初始化
            single_lane_entry_exit_info['right_neighbors_id'] = -1
            if list(single_feature.lane.left_neighbors) != []:  #左边有车道
                left_neighbors_temp = list(single_feature.lane.left_neighbors)
                flag1 = 0
                for left_neighbor in left_neighbors_temp:
                    # print('index')
                    # print(ego_lane_id)
                    # print(lane_index_all)
                    # print(left_neighbor.self_end_index)
                    if abs(left_neighbor.self_end_index - lane_index_all)<2:
                        left_neighbors_id = left_neighbor.feature_id  #记录满足条件的那条左边相邻车道的ID
                        #print("left_neighbors %d" % left_neighbors_id)
                        flag1 = 1
                        break
                if flag1 ==1:
                    single_lane_entry_exit_info['left_neighbors_id'] = left_neighbors_id
                    #print(left_neighbors_id)

            if list(single_feature.lane.right_neighbors) != []:  #右边右车道
                right_neighbors_temp = list(single_feature.lane.right_neighbors)
                flag2 = 0
                for right_neighbor in right_neighbors_temp:
                    #print('index222')
                    #print(lane_index_all)
                    #print(left_neighbor.self_end_index)
                    if abs(right_neighbor.self_end_index -lane_index_all)<2:
                        right_neighbors_id = right_neighbor.feature_id
                        #print("right_neighbors %d"%right_neighbors_id)
                        flag2 = 1
                        break
                if flag2 == 1:
                    single_lane_entry_exit_info['right_neighbors_id'] = right_neighbors_id

            #将一些信息重新记录，便于数据检索和处理
            lane_id_all.append(single_feature.id)
            all_lane_entry_exit_info.append(single_lane_entry_exit_info)
            single_feature_all_lane_polyline.append((single_feature.id, single_feature.lane.polyline))
            single_map_dict[single_feature.id] = single_feature.lane.polyline  # 使用字典进行检索，得到所有车道的坐标点信息
    # print('qqqq')
    # print(single_scenario_all_lane_entry_exit_info)
    # print(single_map_dict,lane_turn_left_id,lane_turn_right_id)
    return all_lane_entry_exit_info,single_map_dict,lane_turn_left_id,lane_turn_right_id

def intersection_angle_cal(df_all_lan_topo_info):
    from sklearn.cluster import KMeans
    intersection_angle = 0
    slope_list = pd.unique(df_all_lan_topo_info[df_all_lan_topo_info['is_a_turn']== 'straight']['lane_slope'].tolist())
    #print('slope_list:')
    #print(slope_list)
    estimator = KMeans(n_clusters=2)  # 构造聚类器
    estimator.fit(np.array(slope_list).reshape(-1, 1))  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    k1 = np.mean(slope_list[label_pred==0])   #这一类斜率的平均值计算
    k2 = np.mean(slope_list[label_pred==1])
    #print('k1:%f,k2:%f'%(k1,k2))
    intersection_angle = math.atan(abs((k2-k1)/(1+k1*k2)))*180/np.pi
    return intersection_angle

class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y
def get_point_order(points):  #得到正确的点的连线顺序，以得到正确的多边形
    points_new = []
    points_new_plus = []  #将交叉口范围稍微扩大
    left_point = {}  #某个点左侧最近的点
    right_point = {}  #某个点右侧最近的点
    for A in points:
        angle_vec = 0
        for B in points:
            for C in points:
                if (A.x!=B.x and A.y!=B.y)and (C.x!=B.x and C.y!=B.y) and (C.x!=A.x and C.y!=A.y):
                    vec_AB = Point(B.x-A.x,B.y-A.y)
                    vec_AC = Point(C.x-A.x,C.y-A.y)
                    vec_BA = Point(-vec_AB.x,-vec_AB.y)
                    #print(vec_AB.x,vec_AB.y,vec_AC.x,vec_AC.y,A.x,A.y,C.x,C.y)
                    #print(abs(math.sqrt(vec_AB.x**2+vec_AB.y**2)*math.sqrt(vec_AC.x**2+vec_AC.y**2)))
                    cos_angle = (vec_AB.x*vec_AC.x+vec_AB.y*vec_AC.y)/abs(math.sqrt(vec_AB.x**2+vec_AB.y**2)*math.sqrt(vec_AC.x**2+vec_AC.y**2))
                    #print(cos_angle)
                    angle_vec_temp = abs(math.acos(cos_angle)*180/np.pi)
                    #print(angle_vec_temp)
                    if angle_vec_temp > angle_vec:
                        angle_vec = angle_vec_temp
                        #print(angle_vec)
                        p5 = vec_BA.x*vec_AC.y - vec_AC.x*vec_BA.y
                        if p5>0:
                            left_point[points.index(A)] = points.index(B)
                            right_point[points.index(A)] = points.index(C)
                        elif p5<0:
                            left_point[points.index(A)] = points.index(C)
                            right_point[points.index(A)] = points.index(B)

    A = points[0]
    points_new.append((A.x,A.y))
    points_new_plus.append((A.x+5,A.y+5))
    points_new_index = [0]
    for i in range(20):
        A = points[left_point[points.index(A)]]
        points_new.append((A.x,A.y))
        points_new_plus.append((A.x + 5, A.y + 5))
        if left_point[points.index(A)] == 0:
            break
        points_new_index.append(left_point[points.index(A)])
    return points_new,points_new_plus


def rayCasting(p, poly):  #判断一个点是否在多边形内部
    px,py = p[0],p[1]
    flag = -1
    i = 0
    l = len(poly)
    j = l - 1
    # for(i = 0, l = poly.length, j = l - 1; i < l; j = i, i++):
    while i < l:
        sx = poly[i][0]
        sy = poly[i][1]
        tx = poly[j][0]
        ty = poly[j][1]
        # 点与多边形顶点重合
        if (sx == px and sy == py) or (tx == px and ty == py):
            flag = 1
        # 判断线段两端点是否在射线两侧
        if (sy < py and ty >= py) or (sy >= py and ty < py):
            # 线段上与射线 Y 坐标相同的点的 X 坐标
            x = sx + (py - sy) * (tx - sx) / (ty - sy)
            # 点在多边形的边上
            if x == px:
                flag = 1
            # 射线穿过多边形的边界
            if x > px:
                flag = -flag
        j = i
        i += 1
    # 射线穿过多边形边界的次数为奇数时点在多边形内
    return flag

def judge_lane_in_intersection(lane_polyline,intersection_range):
    flag = 0  #如果flag=1 则这条车道在交叉口内部，否则不在
    point_start = (lane_polyline[0].x,lane_polyline[0].y)
    point_end = (lane_polyline[-1].x,lane_polyline[-1].y)
    if (rayCasting(point_start, intersection_range) == 1) and (rayCasting(point_end, intersection_range) == 1):
        flag = 1
    return flag

def get_one_direction_lane_info(right_lane_id,df_all_lan_topo_info,single_map_dict,lane_turn_left_id):
    lane_in_num,lane_out_num = 0,0  #这里统计的出口道信息是进口道方向逆时针转90°后方向的出口道
    lane_in_id,lane_out_id = [],[]
    entry_lane_id = df_all_lan_topo_info[df_all_lan_topo_info['lane_id']==right_lane_id]['entry_lanes'].iloc[0][0]
    exit_lane_id = df_all_lan_topo_info[df_all_lan_topo_info['lane_id']==right_lane_id]['exit_lanes'].iloc[0][0]
    #----------------进口道信息提取--------------------
    #车道ID及数量
    #右转车道右侧的车道一般是自行车车道，暂时不做处理
    lane_in_id.append(entry_lane_id)

    lane_in_left = df_all_lan_topo_info[df_all_lan_topo_info['lane_id'] == entry_lane_id]['left_neighbors_id']
    while (lane_in_left.tolist()[0] != -1):#如果左侧没有车道，则为单车道
        flag = 0
        lane_in_left = df_all_lan_topo_info[df_all_lan_topo_info['lane_id'] == entry_lane_id]['left_neighbors_id']
        # print('ssddffs')
        #print(lane_in_left.tolist())
        # print(lane_in_left.tolist())
        # print(lane_in_left.values)
        # print(lane_in_left.tolist()[0] ==-1)
        if (lane_in_left.tolist()[0] ==  -1):
            break
        lane_in_id.append(lane_in_left.tolist()[0])
        entry_lane_id = lane_in_left.tolist()[0]
        #print(entry_lane_id)
    lane_in_num = len(lane_in_id)
    #-------------------出口道信息提取---------------------------
    # 车道ID及数量

    lane_out_id.append(exit_lane_id)
    lane_out_left = df_all_lan_topo_info[df_all_lan_topo_info['lane_id'] == exit_lane_id]['left_neighbors_id']

    while (lane_out_left.tolist()[0] != -1):#如果左侧没有车道，则为单车道
        flag = 0
        lane_out_left = df_all_lan_topo_info[df_all_lan_topo_info['lane_id'] == exit_lane_id]['left_neighbors_id']
        if (lane_out_left.tolist()[0] ==  -1):
            break
        lane_out_id.append(lane_out_left.tolist()[0])
        exit_lane_id = lane_out_left.tolist()[0]
    lane_out_num = len(lane_out_id)

    return lane_in_num, lane_in_id, lane_out_num, lane_out_id

def get_lane_width(df_all_lan_topo_info,single_map_dict,lane_in_num, lane_in_id, lane_out_num, lane_out_id):
    lane_in_width, lane_out_width = 0,0
    #计算进口道车道宽度
    #车道宽度计算
    if lane_in_num>1:
        width_in_sum = 0
        for i in range(lane_in_num-1):
            lane_in_cal_1_id = lane_in_id[i]
            lane_in_cal_2_id = lane_in_id[i + 1]
            x1,y1 = single_map_dict[lane_in_cal_1_id][-1].x, single_map_dict[lane_in_cal_1_id][-1].y   #进口道这里是最后一个点，出口道这里应该第1个点
            x2, y2 = single_map_dict[lane_in_cal_2_id][-1].x, single_map_dict[lane_in_cal_2_id][-1].y
            #print("one lane width is %f"%(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)))
            width_in_sum += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        lane_in_width = width_in_sum / (lane_in_num-1)
        #print("lane width ave = %f"%lane_in_width)
    elif lane_in_num == 1:
        lane_in_width = 3.5  #这里需要统筹对向出口道才行

    #计算出口道车道宽度
    if lane_out_num > 1:
        width_out_sum = 0
        for i in range(lane_out_num-1):
            lane_out_cal_1_id = lane_out_id[i]
            lane_out_cal_2_id = lane_out_id[i+1]
            x1,y1 = single_map_dict[lane_out_cal_1_id][0].x,single_map_dict[lane_out_cal_1_id][0].y
            x2, y2 = single_map_dict[lane_out_cal_2_id][0].x, single_map_dict[lane_out_cal_2_id][0].y
            width_out_sum += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        lane_out_width = width_out_sum / (lane_out_num-1)
    elif lane_out_num ==1:
        lane_out_width = 3.5
    return lane_in_width,lane_out_width

def intersection_info_extract(df_all_lane_topo_info,single_map_dict,lane_turn_left_id_ori,lane_turn_right_id_ori,intersection_center_loc,length_1):

    intersection_info = {}
    intersection_info['file_index'] = df_all_lane_topo_info['file_index'].iloc[0]
    intersection_info['scenario_id'] = df_all_lane_topo_info['scenario_id'].iloc[0]
    intersection_info['intersection_center_point'] = intersection_center_loc  # 交互车辆所在的中心位置
    #---------------------筛选该目标交叉口范围的所有左转、右转车道------------------------
    length_1 = 100  #初步将交叉口范围定在50m，后面可能需要更改
    A = (intersection_center_loc[0]-length_1/2,intersection_center_loc[1]+length_1/2)
    B = (intersection_center_loc[0]+length_1/2,intersection_center_loc[1]+length_1/2)
    C = (intersection_center_loc[0]+length_1/2,intersection_center_loc[1]-length_1/2)
    D = (intersection_center_loc[0]-length_1/2,intersection_center_loc[1]-length_1/2)
    intersection_range_approximate = [A,B,C,D]
    lane_turn_left_id = []
    lane_turn_right_id = []
    #筛选目标交叉口内部的左转、右转车道
    for left_id in lane_turn_left_id_ori:
        if (judge_lane_in_intersection(single_map_dict[left_id],intersection_range_approximate)==1):
            lane_turn_left_id.append(left_id)
    for right_id in lane_turn_right_id_ori:
        if (judge_lane_in_intersection(single_map_dict[right_id],intersection_range_approximate)==1):  #!!!
            lane_turn_right_id.append(right_id)
    # print('mmmm')
    # print(lane_turn_left_id_ori, lane_turn_right_id_ori)
    # print(lane_turn_left_id,lane_turn_right_id)
    intersection_info['lane_id_turn_left_inside'] = lane_turn_left_id
    intersection_info['lane_id_turn_right_inside'] = lane_turn_right_id
    #以下的处理全部基于所有右转、左转车道的分流点、合流点进行
    merging_points_left = []
    merging_points_right = []  #右转车道的合流点
    diverging_points_left = []
    diverging_points_right = []  #右转车道的分流点
    all_lane_id = pd.unique(df_all_lane_topo_info['lane_id'].tolist())

    points_key = []  #右转合流点、分流点集合
    for right_lane_id in lane_turn_right_id:
        #提取所有与右转车道连接的分流点、合流点
        point_start_x = single_map_dict[right_lane_id][0].x  #起点，即分流点
        point_start_y = single_map_dict[right_lane_id][0].y
        point_end_x = single_map_dict[right_lane_id][-1].x  #终点，即合流点
        point_end_y = single_map_dict[right_lane_id][-1].y
        merging_points_right.append((point_end_x,point_end_y))
        diverging_points_right.append((point_start_x,point_start_y))
        point_start = Point(point_start_x,point_start_y)
        point_end = Point(point_end_x,point_end_y)
        points_key.append(point_start)
        points_key.append(point_end)
        #得到该右转车道的进入车道和驶出车道
        #print('test_1:')
        #print(df_single_scenario_lane_topo_info[df_single_scenario_lane_topo_info['lane_id']==right_lane_id]['entry_lanes'])
        entry_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == right_lane_id]['entry_lanes'].iloc[0]
        exit_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == right_lane_id]['exit_lanes'].iloc[0]
        #判断与右转车道相连接的车道是进口道还是出口道
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == entry_lane_id, 'entry_or_exit'] = 'entry'
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == exit_lane_id, 'entry_or_exit'] = 'exit'
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == right_lane_id, 'entry_or_exit'] = 'inside'  #右转车道自身位于交叉口内部
        #将进口道、出口道的车道功能标记一下
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == entry_lane_id, 'lane_function'] = 'right' #记录车道功能，本身是直行车道，功能是用于右转
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == exit_lane_id, 'lane_function'] = 'right'
    df_all_lane_topo_info.loc[:, 'entry_or_exit'] = ''
    #print(points_key,lane_turn_right_id)
    points_key_new, points_key_new_plus = get_point_order(points_key)  # 得到正确的能够将所有右转合流、分流点连接为多边形的顺序  points_key_new_plus 将整个多边形向外扩展5m，以满足冗余
    for lane_in_id in all_lane_id:
        if (lane_in_id not in lane_turn_left_id) and (lane_in_id not in lane_turn_right_id):
            if judge_lane_in_intersection(single_map_dict[lane_in_id], points_key_new_plus) ==1:
                df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == lane_in_id, 'entry_or_exit'] = 'inside'  # 车道位于交叉口内部

    for left_lane_id in lane_turn_left_id: #对于所有的左转车道
        #df_single_scenario_lane_topo_info[df_single_scenario_lane_topo_info['lane_id'] == left_lane_id]['entry_or_exit'] = 'inside'  # 左转车道自身位于位于交叉口内部
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == left_lane_id, 'entry_or_exit'] = 'inside'  # 左转车道自身位于位于交叉口内部
        entry_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == left_lane_id]['entry_lanes'].iloc[0]
        exit_lane_id = df_all_lane_topo_info[df_all_lane_topo_info['lane_id'] == left_lane_id]['exit_lanes'].iloc[0]
        #记录进出口道功能
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == entry_lane_id, 'entry_or_exit'] = 'entry'
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == exit_lane_id, 'entry_or_exit'] = 'exit'
        #标记车道功能
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == entry_lane_id, 'lane_function'] = 'left'  #本身是直行车道，功能是用于左转
        df_all_lane_topo_info.loc[df_all_lane_topo_info['lane_id'] == exit_lane_id, 'lane_function'] = 'left'
    #print('sss')
    #print(df_single_scenario_lane_topo_info['entry_or_exit'])
    intersection_info['intersection_anlge'] = round(intersection_angle_cal(df_all_lane_topo_info), 2)
    #交叉口类型判断，这里需要修正，需要讨论更多的类型
    if len(merging_points_right) >= 4 and len(diverging_points_right)>=4 :
        if 105>=intersection_info['intersection_anlge']>=75:
            intersection_info['Type'] = 'Cross'  #十字型交叉口
        elif 0<intersection_info['intersection_anlge']<75 or 105<intersection_info['intersection_anlge']<180:
            intersection_info['Type'] = 'X'  #X型交叉口
    elif len(merging_points_right) >= 2 and len(diverging_points_right)>= 2 :
        intersection_info['Type'] = 'T'  #T型交叉口

    #----------Extract the lane number and width of the intersection
    extract_direction_index = 1
    #print(lane_turn_right_id)
    for right_lane_id in lane_turn_right_id:
        lane_in_num, lane_in_id, lane_out_num, lane_out_id = get_one_direction_lane_info(right_lane_id, df_all_lane_topo_info, single_map_dict, lane_turn_left_id)
        lane_in_width,lane_out_width = get_lane_width(df_all_lane_topo_info, single_map_dict, lane_in_num, lane_in_id, lane_out_num, lane_out_id) #计算车道宽度
        #print(lane_in_num, lane_in_id, lane_in_width,lane_out_num, lane_out_id, lane_out_width)
        #进口道车道数量、车道id、车道宽度信息记录
        intersection_info['direction_' + str(extract_direction_index) + '_in_lane_num'] = lane_in_num
        intersection_info['direction_' + str(extract_direction_index) + '_in_lane_id_list'] = lane_in_id
        intersection_info['direction_' + str(extract_direction_index) + '_in_lane_width'] = round(lane_in_width,2)
        #出口道车道数量、车道id、车道宽度信息记录
        intersection_info['direction_' + str(extract_direction_index) + '_out_lane_num'] = lane_out_num
        intersection_info['direction_' + str(extract_direction_index) + '_out_lane_id_list'] = lane_out_id
        intersection_info['direction_' + str(extract_direction_index) + '_out_lane_width'] = round(lane_out_width)
        extract_direction_index += 1

    return intersection_info, df_all_lane_topo_info,lane_turn_right_id

def map_lane_point_extract(map_features,file_index,scenario_label):
    map_point_list = []
    for single_feature in map_features:
        if list(single_feature.road_edge.polyline) != []:  #类型为road_edge
            for polyline in single_feature.road_edge.polyline:
                dic_map = {}
                dic_map['file_index'] = file_index
                dic_map['scenario_label'] = scenario_label
                dic_map['line_id'] = single_feature.id
                dic_map['type'] = 'road_edge'
                if single_feature.road_edge.type == 0:
                    dic_map['line_type'] = 'UNKNOWN'
                elif single_feature.road_edge.type ==1:
                    dic_map['line_type'] = 'ROAD_EDGE_BOUNDARY'
                elif single_feature.road_edge.type == 2:
                    dic_map['line_type'] = 'ROAD_EDGE_MEDIAN'
                dic_map['point_x'] = polyline.x
                dic_map['point_y'] = polyline.y
                dic_map['point_z'] = polyline.z
                map_point_list.append(dic_map)
        if list(single_feature.lane.polyline) != []:
            for polyline in single_feature.lane.polyline:
                dic_map = {}
                dic_map['file_index'] = file_index
                dic_map['scenario_label'] = scenario_label
                dic_map['line_id'] = single_feature.id
                dic_map['type'] = 'lane'
                if single_feature.lane.type == 0:
                    dic_map['line_type'] = 'UNDEFINED'
                elif single_feature.lane.type ==1:
                    dic_map['line_type'] = 'FREEWAY'
                elif single_feature.lane.type == 2:
                    dic_map['line_type'] = 'SURFACE_STREET'
                elif single_feature.lane.type == 3:
                    dic_map['line_type'] = 'BIKE_LANE'
                dic_map['point_x'] = polyline.x
                dic_map['point_y'] = polyline.y
                dic_map['point_z'] = polyline.z
                dic_map['entry_lanes'] = single_feature.lane.entry_lanes
                dic_map['exit_lanes'] = single_feature.lane.exit_lanes
                # dic_map['left_neighbors_id'] = single_feature.lane.left_neighbors.feature_id
                # dic_map['right_neighbors_id'] = single_feature.lane.right_neighbors.feature_id
                map_point_list.append(dic_map)

        if list(single_feature.road_line.polyline) != []:
            for polyline in single_feature.road_line.polyline:
                dic_map = {}
                dic_map['file_index'] = file_index
                dic_map['scenario_label'] = scenario_label
                dic_map['line_id'] = single_feature.id
                dic_map['type'] = 'road_line'
                if single_feature.road_line.type == 0:
                    dic_map['line_type'] = 'UNKNOWN'
                elif single_feature.road_line.type ==1:
                    dic_map['line_type'] = 'BROKEN_SINGLE_WHITE'
                elif single_feature.road_line.type == 2:
                    dic_map['line_type'] = 'SOLID_SINGLE_WHITE'
                elif single_feature.road_line.type == 3:
                    dic_map['line_type'] = 'SOLID_DOUBLE_WHITE'
                elif single_feature.road_line.type == 4:
                    dic_map['line_type'] = 'BROKEN_SINGLE_YELLOW'
                elif single_feature.road_line.type == 5:
                    dic_map['line_type'] = 'BROKEN_DOUBLE_YELLOW'
                elif single_feature.road_line.type == 6:
                    dic_map['line_type'] = 'SOLID_SINGLE_YELLOW'
                elif single_feature.road_line.type == 7:
                    dic_map['line_type'] = 'SOLID_DOUBLE_YELLOW'
                elif single_feature.road_line.type == 8:
                    dic_map['line_type'] = 'PASSING_DOUBLE_YELLOW'
                dic_map['point_x'] = polyline.x
                dic_map['point_y'] = polyline.y
                dic_map['point_z'] = polyline.z
                map_point_list.append(dic_map)
    df_single_scenario_map_point = pd.DataFrame(map_point_list)
    return df_single_scenario_map_point


if __name__ == '__main__':
    #-----------------------load_data ---------------------
    # file_path_1 = 'D:/Data/'
    # file_path_2 = 'WaymoData_motion_1'
    # file_path_3 = '/training_20s.tfrecord-*-of-01000'
    test_state = 0
    filepath_oridata = 'E:/waymo_motion_dataset/training_20s.tfrecord-*-of-01000'
    all_file_list, file_index_list = get_file_list(filepath_oridata)
    scenario_all_count = 0
    all_intersection_info = []
    target_scenario = 18
    length = 80   #目标交叉口的距离范围，需要进行调整
    for i in tqdm(range(len(file_index_list))):
        single_segment_all_scenario_all_lane_topo_info = []
        file_index = file_index_list[i]
        segment_file = all_file_list[i]
        print('Now is the file:%s' % file_index)
        # filepath_trj = 'E:/Result_save/data_save/all_scenario_all_objects_info/' + file_index + '_all_scenario_all_object_info_1.csv'
        # seg_trj = pd.read_csv(filepath_trj)
        # single_seg_all_scenario_id = pd.unique(seg_trj['scenario_label'])
        segment_name_list = []
        segment_dataset = tf.data.TFRecordDataset(segment_file)
        segment_dataset = segment_dataset.apply(tf.data.experimental.ignore_errors())

        scenario_label = 0  # 所有场景的ID数量记录
        df_single_seg_all_scenario_lane_topo_info = pd.DataFrame()
        df_single_seg_all_scenario_map_point_info = pd.DataFrame()  #静态地图所有车道线的散点信息提取
        #single_file_all_scenario = []
        for one_scenario in segment_dataset:  # one_scenario 就是一个scenario
            single_scenario_all_feature = []
            scenario_label += 1
            scenario_all_count += 1
            #print('Now is the scenario:%s' % scenario_label)
            if test_state == 1:
                #intersection_center_loc = (1798.4981814285616, -2136.2457659217735)
                if scenario_label != target_scenario:
                    continue
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

            map_features_id_list = []
            # scenario_trj = seg_trj[seg_trj['scenario_label'] == scenario_label]

            df_single_scenario_map_point = map_lane_point_extract(map_features,file_index,scenario_label)  #提取地图中所有车道的散点信息
            df_single_seg_all_scenario_map_point_info = pd.concat([df_single_seg_all_scenario_map_point_info,df_single_scenario_map_point],axis=0)#纵向方向合并df



        #-----------------ouput------------------

        outpath_map_point = 'D:/Result_save/data_save/static_map_point_info/' + file_index + '_all_scenario_static_map_info.csv'
        df_single_seg_all_scenario_map_point_info.to_csv(outpath_map_point)




