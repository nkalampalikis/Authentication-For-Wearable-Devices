# Import Statements
import numpy as np
from db import DB
import os
import json
from params import Parameters


params = Parameters()

#####################################################################################
# Function: load_from_db
# param: who - integer ID of the patient who's data we you want to fetch
# param: window_sz - integer Number of seconds of data in each data segment
# param: session_id - integer ID of the training session for the given patient
# param: sequence_id - integer ID of the sequence within the given training session
# returns: A list of segments that represents all the data for the particular sequence
#          each segment is of the form [[accl_x],[accl_y], [accl_z], [gyro_x], [gyro_y], [gyro_z]]
#          where the number of data points in each sub array is the equal to window_sz * 50
#####################################################################################
def load_from_db(who, window_sz, session_id, sequence_id):

    dir_path = os.path.dirname(os.path.realpath(__file__))

    #print("Loading From DB: " + dir_path + "/databases_BVP/db" + "_" + str(window_sz) + ".sqlite")
    #db = DB(dir_path + "/databases_BCG/db" + "_" + str(window_sz) + ".sqlite", init=False)
    db = params.db
    #DB(dir_path + "/databases_BVP/db" + "_" + str(window_sz) + ".sqlite", init=False)

    tables = db.cursor.execute("""
        SELECT table_name FROM data
        WHERE subject_id = ? AND window_sz = ? AND session_id = ?
        AND sequence_id = ?;
    """, [who, window_sz, session_id, sequence_id])

    t = tables.fetchone()

    records = db.cursor.execute(
        '''
            SELECT
                segment_gyro_x,segment_gyro_y,segment_gyro_z,
                segment_accl_x,segment_accl_y,segment_accl_z
            FROM `'''
        + t[0] + '`;')
    records = records.fetchall()

    r = []
    for record in records:
        r.append(record)
    return r


#####################################################################################
# Function: extract_tensor
# param: segment - array of arrays of floats segment data of the form
#                  [[accl_x], [accl_y], [accl_z], [gyro_x], [gyro_y], [gyro_z]]
#                  each sub-array is has window_sz * hz pieces of data, as floats
# param: window_sz - integer number of seconds of data in each data segment
# param: hz - integer frequency of the data collection (50 in our case)
# returns:
#####################################################################################
def extract_tensor(segment, window_sz, hz):
    gyro_layer = []
    accel_layer = []
    for accel_seg in segment[:3]:
        bvp = np.array(accel_seg).reshape(1, 1, 1, window_sz*hz)
        if len(accel_layer) == 0:
            accel_layer = bvp
        else:
            accel_layer = np.concatenate((accel_layer, bvp), axis=2)
    for gyro_seg in segment[3:]:
        bvp = np.array(gyro_seg).reshape(1, 1, 1, window_sz*hz)
        if len(gyro_layer) == 0:
            gyro_layer = bvp
        else:
            gyro_layer = np.concatenate((gyro_layer, bvp), axis=2)
    return np.concatenate((gyro_layer, accel_layer), axis=1)


#####################################################################################
# Function: sequence_to_nparray
# param: sequence -
# param: window_sz -
# param: hz -
# returns:
#####################################################################################
def sequence_to_nparray(sequence, window_sz, hz):
    list_nparrays = []

    for segment in sequence:
        # print (segment)
        # for sad in segment:
        #     print (sad)
        s = []
        for channel in segment:
            c = json.loads(channel)
            s.append(c[0:window_sz*hz])
        numpy_array = extract_tensor(s, window_sz, hz)
        list_nparrays.append(numpy_array)

    return list_nparrays


#####################################################################################
# Function: collect_segment_data
# param: who - integer ID of the patient who's data we you want to fetch
# param: window_sz - integer Number of seconds of data in each data segment
# param: session_id - integer ID of the training session for the given patient
# param: sequence_id - integer ID of the sequence within the given training session
# returns:
#####################################################################################
def collect_segment_data(params, person, train_segs):
    window_sz = params.window_sz
    hz = params.hz
    points = []
    #print("Loading From DB: /databases_BVP/db" + "_" + str(window_sz) + ".sqlite")
    for target in train_segs:
        sequence = load_from_db(person, window_sz, target[0], target[1])
        segs_nparrays = sequence_to_nparray(sequence, window_sz, hz)

        points += segs_nparrays
    return points


# From a list, create a set of 1-overlapped arrays of $sequence_length length
# e.g. [1,2,3,4,5] -> [[1,2,3],[2,3,4],[3,4,5]] with $sequence_length
# Sequence_length is the number of data points per sub array, that overlap with each other by 1 data point
# this increases the training accuracy for the CNN
# Window_size is the number of data points submitted to the cnn divided by the frequency
def make_sequential(params, data, sequence_length):
    out = []
    segcnt = 0
    for i in range(0, len(data)-sequence_length):
        seq_tensor = np.ndarray((
            sequence_length,
            2,
            3,
            params.window_sz*params.hz
        ))

        for k in range(0,sequence_length):
            seq_tensor[k] = data[i+k]
        out.append(seq_tensor)
    return out


# # TESTING
# seq  = load_from_db(28,3,1,1)
# index = 0
#
# nparrays_res = sequence_to_nparray(seq, 3, 50)
#
# print (nparrays_res[0].shape)


# ax_array = divide_array("ax.csv", 3, 50)
# ay_array = divide_array("ay.csv", 3, 50)
# az_array = divide_array("az.csv", 3, 50)
# gx_array = divide_array("gx.csv", 3, 50)
# gy_array = divide_array("gy.csv", 3, 50)
# gz_array = divide_array("gz.csv", 3, 50)
#
# list_nparrays = array_to_nparray(ax_array, ay_array, az_array, gx_array, gy_array, gz_array, 3, 50)
#
# print(list_nparrays[0].shape)