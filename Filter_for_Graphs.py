import numpy as np
import csv
import os
from params import Parameters
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter
from scipy.stats import zscore
import pandas as pd

targets = list(range(1, 29))
param = Parameters(5)
hz = 50

class BVP:
    VERSION = 1.0
    def __init__( self, raw, hz):
        self.hz = hz

        # Set up intermediate arrays
        data_zscore = {}
        data_norm = {}
        data_avg = {}
        data_butter = {}
        self.bvp_y = {}

        # For each component (gyro_{x,y,z}, accel_{x,y,z})
        #for v in list(filter((lambda x: x != "TS"), raw.dtype.names)):
        #for v in range(len(raw)):
        data_zscore = zscore(raw)

        data_avg = self._rolling_avg_filter(data_zscore,3)

        data_butter = self._butter_bandpass_filter(data_avg,10,13,hz,4)

        data_butter = self._butter_bandpass_filter(data_avg, 0.75, 2.5, hz, 2)

        data_norm = self._normalize(data_butter)

        self.bvp_y = self._clean(data_norm)

    def _clean( self, data):
        return data

    def _normalize( self, data ):
        dmax = max(data)
        dmin = min(data)

        # print("dmax = %d, dmin = %d" % (dmax, dmin))
        for i in range(0,len(data)-1):
            data[i] = (data[i]-dmin)/(dmax-dmin)

        return data

    @staticmethod
    def _average_filter( data, size=5 ):
        if( size % 2 == 1 ):
                size += 1
        half = int( size / 2 )
        l = len(data)
        avg = [0]*l

        # Mirror the head and tail
        head = data[0:half]
        tail = data[-1*half:]
        extended_data = np.concatenate( [head[::-1], data, tail[::-1]] )
        for i in range(0, l ):
            avg[i] = np.sum( extended_data[i:i+size] ) / size
        return avg

    @staticmethod
    def _rolling_avg_filter( data, size=5 ):
        avg = BVP._average_filter( data, size )
        return [ x[0] - x[1] for x in zip( data, avg ) ]

    @staticmethod
    def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='bandpass')
        y = lfilter(b, a, data)
        return y



def _align( start_time, data, hz ):
    # Figure out how many sample fall out of range and prune them
    end_time = data["TS"][-1]
    samples = 0
    for r in data["TS"]:
        if( r >= start_time ):
            break
        samples += 1
    if( samples > 0 ):
        data = np.split( data, [ samples - 1 ] )[ 1 ]


    # Check if TS has duplicates and remove if necessary
    s = np.sort(data["TS"], axis=None)
    kill_idexes = []
    for d in s[:-1][s[1:] == s[:-1]]:
        dups = np.where(data["TS"] == d)[0]
        for i in range(0, len(dups)-1):
            kill_idexes.append(dups[i])
    ts = np.delete(data["TS"], kill_idexes)
    x = np.delete(data["X"], kill_idexes)
    y = np.delete(data["Y"], kill_idexes)
    z = np.delete(data["Z"], kill_idexes)

    # Interpolate so samples are evenly spaced
    r = np.arange(start_time, end_time, int(1000 / hz))
    xinterp = (interp1d( ts, x ))(r)
    yinterp = (interp1d( ts, y ))(r)
    zinterp = (interp1d( ts, z ))(r)
    array = []
    for i in range(0, len(r)):
        TS = float( r[i] )
        X = float( xinterp[i] )
        Y = float( yinterp[i] )
        Z = float( zinterp[i] )
        array.append( ( TS, X, Y, Z ) )
    dtype = [
                ('TS', 'float64'),
                ('X', 'float64'),
                ('Y', 'float64'),
                ('Z', 'float64')
            ]
    data = np.array(array, dtype)
    return data


def _synchronize(data_a, data_g, hz):
    # Identify common start point
    g_start = data_g[ "TS" ][ 0 ]
    a_start = data_a[ "TS" ][ 0 ]
    offset = g_start - a_start

    # gyro starts before accel
    if( offset < 0 ):
        #print( "=> Aligning" )
        data_g = _align( data_a["TS"][0], data_g, hz )
        data_a = _align( data_a["TS"][0], data_a, hz )
    # accel starts before gyro
    else:
        data_a = _align( data_g["TS"][0], data_a, hz )
        data_g = _align( data_g["TS"][0], data_g, hz )

    # Force length conformity
    g_len = len( data_g )
    a_len = len( data_a )
    diff = g_len - a_len

    # Gyro is longer
    if( g_len > a_len ):
        data_g = np.split( data_g, [ a_len ] )[ 0 ]
    # Accel is longer
    elif( a_len > g_len ):
        data_a = np.split( data_a, [ g_len ] )[ 0 ]
    return data_a, data_g

def load(gyro_file, accel_file, subject_id, session_id, sequence_id, window_sz, hz):
    f_gyro = open(gyro_file, "r" )
    f_accel = open(accel_file, "r")
    reader_gyro = csv.DictReader( f_gyro )
    reader_accel = csv.DictReader( f_accel )

    def reader_to_array(reader):
        output = []
        for row in reader:
            TS = int( row["TS"] )
            X = float( row["X"] )
            Y = float( row["Y"] )
            Z = float( row["Z"] )
            output.append( ( TS, X, Y, Z ) )


        dtype = [('TS', 'int64'), ('X', 'float64'), ('Y', 'float64'), ('Z', 'float64')]
        output = np.array(output, dtype)
        output = np.sort(output, order='TS', kind="mergesort")

        return output

    raw_gyro = reader_to_array(reader_gyro)
    raw_accel = reader_to_array(reader_accel)

    raw_accel, raw_gyro = _synchronize(raw_accel, raw_gyro, hz)

    TS = []
    raw_accelX = []
    raw_accelY = []
    raw_accelZ = []
    raw_gyroX = []
    raw_gyroY = []
    raw_gyroZ = []

    for a in raw_accel:
        TS.append(a[0])
        raw_accelX.append(a[1])
        raw_accelY.append(a[2])
        raw_accelZ.append(a[3])

    for g in raw_gyro:
        raw_gyroX.append(g[1])
        raw_gyroY.append(g[2])
        raw_gyroZ.append(g[3])


    filtered_AX = BVP(raw_accelX, 50).bvp_y.tolist()
    filtered_AY = BVP(raw_accelY, 50).bvp_y.tolist()
    filtered_AZ = BVP(raw_accelZ, 50).bvp_y.tolist()
    filtered_GX = BVP(raw_gyroX, 50).bvp_y.tolist()
    filtered_GY = BVP(raw_gyroY, 50).bvp_y.tolist()
    filtered_GZ = BVP(raw_gyroZ, 50).bvp_y.tolist()

    graph = "./graphing_csv/{}_{}_{}.csv".format(subject_id, session_id, sequence_id)

    dataframe = pd.DataFrame({'TS': TS, 'AX': filtered_AX, 'AY': filtered_AY, 'AZ': filtered_AZ, 'GX': filtered_GX, 'GY': filtered_GY, 'GZ': filtered_GZ})
    dataframe.to_csv(graph)

    # print(filtered_AX)
    # with open(graph, 'w') as myfile:
    #     wr = csv.writer(myfile)
    #     wr.writerows(['TS'])
    #     wr.writerows(['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ'])
    #     #for val in range(len(filtered_AX)):
    #     wr.writerows([filtered_AX, filtered_AY, filtered_AZ, filtered_GX, filtered_GY, filtered_GZ])





    # # Creates overlaps in a normal array
    # a = []
    # g = []
    # # Use overlapped data
    # # r = list(range(0, window_sz*hz))
    # r = list(range(0, hz))
    # while len(raw_gyro) > 0 and len(raw_accel) > 0:
    #     g.append(raw_gyro[:hz*window_sz])
    #     a.append(raw_accel[:hz*window_sz])
    #     raw_accel = np.delete(raw_accel,r,0)
    #     raw_gyro = np.delete(raw_gyro,r,0)
    #
    # # Drop last segment, as it may not be a full length one
    # max_seg = min(len(a), len(g))
    # segments = []
    # for i in range(0, max_seg):
    #     segments.append(Data(a[i], g[i], hz))
    # segments = list(filter(lambda x: len(x.data) == window_sz*hz, segments))
    #
    #
    # # For each segment,generate a BVP
    # bvps = []
    # print("\rCreating BVP -/{}".format(len(segments)), end='')
    # for d in segments:
    #     print("\rCreating BVP {}/{}".format(len(bvps) + 1, len(segments)), end='')
    #     assert(len(d.data) == window_sz*hz)
    #     b = BVP(d.data,hz)
    #     bvps.append(b)
    # print()


if not os.path.isdir("./graphing_csv/"):
    print("Creating Graphoing Directory")
    os.mkdir("./graphing_csv/")

#Window Size
for t in [param.window_sz]:
    # Participant
    for i in targets:
        # Session
        for j in range(1, 5):
            # Sequence
            for k in range(1, 6):
                g_file = "./data/{}/{}/{}/gyro.csv".format(i, j, k)
                a_file = "./data/{}/{}/{}/accel.csv".format(i, j, k)

                if os.path.isfile(g_file) and os.path.isfile(a_file):
                    print("Loading %d-%d-%d" % (i, j, k))

                    # preprocessed_data = Preprocessor(a_file, g_file, window_size)
                    # segments = preprocessed_data.list_segments_dfs
                    # db.load_to_db(segments, str(i), j, k, t, 50)

                    load(g_file, a_file, str(i), j, k, t, 50)
                else:
                    # print("Skipping %d-%d-%d (does not exist)" % (i,j,k))
                    pass