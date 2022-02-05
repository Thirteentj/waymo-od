import os
import tensorflow as tf
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import Scenario_extract as SExt

if __name__ == '__main__':

    filepath1 = 'E:/Result_save/data_save/all_scenario_all_objects_info/' + '*_all_scenario_all_object_info_1.csv'
    filepath2 = 'E:/Result_save/data_save/objects_of_interest_info/' + '*_all_scenario_objects_of_interest_info.csv'
    data_file_list, data_file_index_list = SExt.get_file_list(filepath1)
    interest_od_file_list, interest_od_index_list = SExt.get_file_list(filepath2)
    #print(data_file_index_list)

    if data_file_index_list != interest_od_index_list:
        print("The file isn't match")
    else:
        print("Data has been loaded")
    SExt.scenario_extract_straight_turn_left(data_file_list, data_file_index_list, interest_od_file_list)
    #SExt.scenario_extract_only_left(data_file_list, data_file_index_list, interest_od_file_list)

