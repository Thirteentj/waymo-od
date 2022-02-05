import os
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

def data_heading_check(df):  #异常数据处理  数据平滑
    length = df.shape[0]
    for i in range(1,length-1):
        if (abs(df['heading_180'].iloc[i] - df['heading_180'].iloc[i-1]) > 30) and (abs(df['heading_180'].iloc[i] - df['heading_180'].iloc[i+1]) > 30):
            df['heading_180'].iloc[i] = (df['heading_180'].iloc[i-1]+df['heading_180'].iloc[i+1])/2

        if (df['heading_180'].iloc[i] - df['heading_180'].iloc[i-1])>350:
            df['heading_180'].iloc[i] -= 360
        if (df['heading_180'].iloc[i] - df['heading_180'].iloc[i-1])<-350:
            df['heading_180'].iloc[i] += 360
    if (df['heading_180'].iloc[length-1] - df['heading_180'].iloc[length - 2]) > 350:
        df['heading_180'].iloc[length-1] -= 360
    if (df['heading_180'].iloc[length-1] - df['heading_180'].iloc[length - 2]) < -350:
        df['heading_180'].iloc[length-1] += 360
    return df
def process_data(df,interest_od):
    df_veh1 = df[df['obj_id'] == interest_od[0]]
    df_veh1['heading_180'] = df_veh1['heading'] * 180 / np.pi  # 将航向角从弧度制转为角度制
    df_veh1 = data_heading_check(df_veh1)
    length1 = df_veh1.shape[0]  # 数据长度
    df_veh2 = df[df['obj_id'] == interest_od[1]]
    #df_veh2['heading_180_0'] = df_veh2['heading'] * 180 / np.pi
    df_veh2['heading_180'] = df_veh2['heading'] * 180 / np.pi  # 将航向角从弧度制转为角度制
    df_veh2 = data_heading_check(df_veh2)
    length2 = df_veh2.shape[0]  # 数据长度
    # 寻找共同的时间区间
    t_begin = max(df_veh1['time_stamp'].min(), df_veh2['time_stamp'].min())
    t_end = min(df_veh1['time_stamp'].max(), df_veh2['time_stamp'].max())  # 取结束时间的较小值
    #print("time_range {},{}".format(t_begin,t_end))
    df_veh1 = df_veh1[(df_veh1['time_stamp'] >= t_begin) & (df_veh1['time_stamp'] <= t_end)]
    df_veh2 = df_veh2[(df_veh2['time_stamp'] >= t_begin) & (df_veh2['time_stamp'] <= t_end)]

    return df_veh1,df_veh2

def get_interest_od(df_obj):
    interest_dict = {}
    scenario_label_list = df_obj['scenario_label'].tolist()
    for label in scenario_label_list:
        #print(df_obj[df_obj['scenario_label']==label].objects_of_interest.tolist()[0])
        interest_dict[label] = eval(df_obj[df_obj['scenario_label']==label].objects_of_interest.tolist()[0])
    return interest_dict
def judge_dis(df_veh1,df_veh2):
    flag = 0  #判断相对距离是否小于距离阈值
    dis_each = []
    length = min(df_veh1.shape[0],df_veh2.shape[0]) #数据长度,即为时间戳的长度
    #print('length {},{}'.format(df_veh1.shape[0],df_veh2.shape[0]))
    min_dis = 9999
    time_stamp_min_dis = -1  #距离最近时的时刻
    time_min_list_temp = []   #用于记录可能的最近记录对应的时刻
    time_range_interactive = ()  #用于记录车辆交互行为的起、终点时刻
    time_inter_list = []   #用于记录车辆交互的时间范围，在距离阈值内时刻都会记录下来
    loc_list_temp = []
    loc_veh_1_min_dis = ()  #用于记录距离最近时对应的两辆车的坐标位置 2辆车*2维信息
    loc_veh_2_min_dis = ()
    #print(df_veh1)
    for i in range(length):
        x1,y1 = df_veh1['center_x'].iloc[i],df_veh1['center_y'].iloc[i]
        x2,y2 = df_veh2['center_x'].iloc[i], df_veh2['center_y'].iloc[i]
        dis = np.sqrt(pow((x1-x2),2)+pow((y1-y2),2))

        if dis < 15:  #小于15m即认为满足距离阈值限制
            flag = 1
            dis_each.append(dis)
            time_min_list_temp.append(float(df_veh1['time_stamp'].iloc[i]))
            loc_veh1 = (x1,y1)
            loc_veh2 = (x2,y2)
            loc_list_temp.append((loc_veh1,loc_veh2))
        if dis < 50:
            time_inter_list.append(float(df_veh1['time_stamp'].iloc[i]))
    if dis_each:
        min_dis = min(dis_each)
        min_dis_index = dis_each.index(min_dis)
        time_stamp_min_dis = time_min_list_temp[min_dis_index]
        loc_veh_1_min_dis, loc_veh_2_min_dis = loc_list_temp[min_dis_index]
    else:
        min_dis = 9999
    time_range_interactive = (min(time_inter_list),max(time_inter_list))   #得到这个列表内的时间最大值和最小值，即为这段交互的整个时间段

    return flag,min_dis,time_stamp_min_dis,time_range_interactive,loc_veh_1_min_dis,loc_veh_2_min_dis

def judge_dis_1(df_veh1,df_veh2):
    flag = 0  #判断相对距离是否小于距离阈值
    dis_each = []
    length = min(df_veh1.shape[0],df_veh2.shape[0]) #数据长度
    #print('length {},{}'.format(df_veh1.shape[0],df_veh2.shape[0]))
    for i in range(length):
        x1,y1 = df_veh1['center_x'].iloc[i],df_veh1['center_y'].iloc[i]
        x2,y2 = df_veh2['center_x'].iloc[i], df_veh2['center_y'].iloc[i]
        dis = np.sqrt(pow((x1-x2),2)+pow((y1-y2),2))
        dis_each.append(dis)
        if dis < 15:
            flag = 1
    if dis_each:
        return flag,min(dis_each)
    else:
        return flag,9999

def judge_angle(df_veh1,df_veh2):
    #如果是-的，加360
    flag = 0  #判断航向角转向是否小于角度阈值
    left_flag = 0 #判断是左转还是右转，左转flag为1
    angle_range1 = float(df_veh1['heading_180'].max() - df_veh1['heading_180'].min())
    index_1 = [df_veh1['heading_180'].idxmax(),df_veh1['heading_180'].idxmin()]   #idxmax()  返回最大值对应的索引
    angle_range2 = float(df_veh2['heading_180'].max() - df_veh2['heading_180'].min())
    index_2 = [df_veh2['heading_180'].idxmax(),df_veh2['heading_180'].idxmin()]
    max_index = 0
    max_index_list = []
    if angle_range1 > angle_range2:
        max_angle = angle_range1
        max_index_list = index_1
        min_angle = angle_range2
        left_veh_id = df_veh1['obj_id'].iloc[0]
    else:
        max_angle = angle_range2
        max_index_list = index_2
        min_angle = angle_range1
        left_veh_id = df_veh2['obj_id'].iloc[0]
    #print(max_index_list)
    #print(max_angle,min_angle)
    if max_angle > 70 and min_angle <40:
        flag = 1
        if max_index_list[0]>max_index_list[1]:
            left_flag = 1
    return flag,left_flag,max_index_list,angle_range1,angle_range2,left_veh_id



def get_file_list(filepath):

    all_files = sorted(glob.glob(filepath))
    segs_name_index = []
    for file in all_files:
        segment_name = os.path.basename(file)
        segs_name_index.append(segment_name[:5])
    #print(segs_name_all)
    #print(all_files)
    return all_files,segs_name_index

if __name__ == '__main__':

    filepath1 = '../Result_save/data_save/all_scenario_all_objects_info/' +'*_all_scenario_all_object_info_*.csv'
    filepath2 = '../Result_save/data_save/objects_of_interest_info/' +'*_all_scenario_objects_of_interest_info.csv'
    data_file_list, data_file_index_list = get_file_list(filepath1)
    interest_od_file_list,interest_od_index_list = get_file_list(filepath2)
    print(data_file_index_list)
    all_left_scenario = []
    if data_file_index_list != interest_od_index_list:
        print("The file isn't match")
    for i in tqdm(range(len(data_file_list))):
        segment_id = data_file_index_list[i]
        #print(data_file_list[i],interest_od_file_list[i])

        df_trj = pd.read_csv(data_file_list[i])
    
        #必须要求是机动车，其次为想办法找到左转场景  右转航向角减小，右转航向角增大
        #print(interest_od_file_list[i])
        df_obj = pd.read_csv(interest_od_file_list[i])
    
        scenario_label_list = list(set(df_trj['scenario_label'].tolist()))
        #print(scenario_label_list)
        interest_od_dict = get_interest_od(df_obj)
        #print('Data has been loaded')
        dic_heading = {}
        for i in range(len(scenario_label_list)):
            try:
                dic_single_left_scenario = {}
                flag_dis = 0
                flag_angle = 0
                label = scenario_label_list[i]
                interest_od = interest_od_dict[label]
                df = df_trj[(df_trj['scenario_label']==label) & (df_trj['valid']==True)]
                #print(df.shape)
                df_veh1, df_veh2 = process_data(df, interest_od)
                if df_veh1['obj_type'].iloc[0] == 1 and df_veh2['obj_type'].iloc[0] == 1:
                    flag_dis, dis_each_min,time_stamp_min_dis,time_range_interactive,loc_veh_1_min_dis,loc_veh_2_min_dis = judge_dis(df_veh1,df_veh2)
                    flag_angle,left_flag,angle_index_list,angle_veh_1,angle_veh_2,left_veh_id = judge_angle(df_veh1,df_veh2)
                    #print("Scenario {:.2f}, Angle is {:.2f} and {:.2f},flag_angle is {} ,distence is {:.2f},angle_index {},left_flag {}".format(label,angle_veh_1,angle_veh_2,flag_angle,dis_each_min,angle_index_list,left_flag))

                    #test
                    heading_1 = 'veh1_' + str(interest_od[0])+'_'+str(label)
                    heading_2 = 'veh2_' + str(interest_od[1])+'_'+str(label)
                    dic_heading[heading_1] = df_veh1['heading_180'].tolist()
                    dic_heading[heading_2] = df_veh2['heading_180'].tolist()

                    if flag_dis == 1 and flag_angle == 1 and left_flag == 1:
                        dic_single_left_scenario['segment_id'] = segment_id
                        dic_single_left_scenario['scenario_id'] = label
                        dic_single_left_scenario['interactive_veh_1_id'] = interest_od[0]
                        dic_single_left_scenario['interactive_veh_2_id'] = interest_od[1]
                        dic_single_left_scenario['turn_left_veh_id'] = left_veh_id
                        dic_single_left_scenario['min_dis'] = dis_each_min
                        dic_single_left_scenario['time_stamp_min_dis'] = time_stamp_min_dis
                        dic_single_left_scenario['time_range_interactive'] = time_range_interactive
                        dic_single_left_scenario['loc_veh_1_interactive'] = loc_veh_1_min_dis
                        dic_single_left_scenario['loc_veh_2_interactive'] = loc_veh_2_min_dis
                        dic_single_left_scenario['intersection_center_loc'] = ((loc_veh_1_min_dis[0]+loc_veh_2_min_dis[0])/2,(loc_veh_1_min_dis[1]+loc_veh_2_min_dis[1])/2)  #用两车的位置的平均值近似代替交叉口的中心位置
                        dic_single_left_scenario['angle_veh_1'] = angle_veh_1
                        dic_single_left_scenario['angle_veh_2'] = angle_veh_2
                        all_left_scenario.append(dic_single_left_scenario)
                        #print('存在左转场景第{}个,两车OD为{}、{},最小距离为{:.2f}m,两车是航向角变化分别为{:.2f}°,{:.2f}°,left_veh_id {}'.format(label,interest_od[0],interest_od[1],dis_each_min,angle_veh_1,angle_veh_2,left_veh_id))
            except:
                continue

    #outpath = 'data/heading_dataV3.xlsx'
    outpath = '../Result_save/Turn_left_scenario_all.csv'
    df_out = pd.DataFrame(all_left_scenario,index=None)
    df_out.to_csv(outpath)
