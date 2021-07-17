
from scipy.stats import zscore
import numpy as np
from scipy.signal import butter, lfilter

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
        for v in list(filter((lambda x: x != "TS"), raw.dtype.names)):
            data_zscore[v] = zscore(raw[v])

            ######FOR BVP #############
            #data_avg[v] = self._rolling_avg_filter(data_zscore[v],3)
            #data_butter[v] = self._butter_bandpass_filter(data_avg[v],10,13,hz,4)
            #data_butter[v] = self._butter_bandpass_filter(data_avg[v], 0.75, 2.5, hz, 2)

            ######FOR BCG #############
            data_avg[v] = self._rolling_avg_filter(data_zscore[v], 35)
            data_butter[v] = self._butter_bandpass_filter(data_avg[v], 4, 11, hz, 4)
            data_norm[v] = self._normalize(data_butter[v])

            self.bvp_y[v] = self._clean(data_norm[v])

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

