import pandas as pd
import matplotlib.pyplot as plt
from bvp_filter import overlap, n_to_bvp_signal, n_to_bvp_signal_overlap
from train import Trainer
from copy import deepcopy
from params import Parameters



################################################################
# MAIN
################################################################

#Extract table from the database

targets = [9, 10, 11, 12, 14, 24]
# targets = [2,3,4,5,7,8,13,15,16,19,26,28]


def train(targets, window_sz):
    p = Parameters()
    p.window_sz = window_sz
    for owner in targets:
        others = deepcopy(targets)
        others.remove(owner)
        t = Trainer(p, owner, others)
        t.train(100)


train(targets, 4)

#
# def main():
#     train(targets, 3)
#
# if __name__ == '__main__':
#     from train import Trainer
#     main()

# hz = 50
# window_sz
#
# # Name the columns
# acceldf = pd.read_csv('testing_data/accel.csv', names=['Time', 'Ax', 'Ay', 'Az'])
# acceldf = acceldf.iloc[1:]
#
# gyrodf = pd.read_csv('testing_data/gyro.csv', names=['Time', 'Gx', 'Gy', 'Gz'])
# gyrodf = gyrodf.iloc[1:]
#
# accel_gyro_df = acceldf.join(gyrodf.set_index('Time'), on='Time').dropna()
#
# time = accel_gyro_df.Time.values
# # Keeping the time data for plotting purposes
# df_AX = accel_gyro_df.Ax
# df_AY = accel_gyro_df.Ay
# df_AZ = accel_gyro_df.Az
# df_GX = accel_gyro_df.Gx
# df_GY = accel_gyro_df.Gy
# df_GZ = accel_gyro_df.Gz
#
#
# # Strip the time for CNN input preparation
# df_AX_overlapped = [overlap(accel_gyro_df.Ax.values, 3)]
# df_AY_overlapped = [overlap(accel_gyro_df.Ay.values, 3)]
# df_AZ_overlapped = [overlap(accel_gyro_df.Az.values, 3)]
# df_GX_overlapped = [overlap(accel_gyro_df.Gx.values, 3)]
# df_GY_overlapped = [overlap(accel_gyro_df.Gy.values, 3)]
# df_GZ_overlapped = [overlap(accel_gyro_df.Gz.values, 3)]
#
# df_AX_overlapped = n_to_bvp_signal_overlap(df_AX_overlapped,lower_limit=0, upper_limit=len(df_AX_overlapped))
# print(df_AX_overlapped)
#
# # Process all the sensor axes into data frames that represent a BVP signal
# ax_df = n_to_bvp_signal([df_AX], time, lower_limit=100, upper_limit00)
# ay_df = n_to_bvp_signal([df_AY], time, lower_limit=100, upper_limit00)
# az_df = n_to_bvp_signal([df_AZ], time, lower_limit=100, upper_limit00)
# gx_df = n_to_bvp_signal([df_GX], time, lower_limit=100, upper_limit00)
# gy_df = n_to_bvp_signal([df_GY], time, lower_limit=100, upper_limit00)
# gz_df = n_to_bvp_signal([df_GZ], time, lower_limit=100, upper_limit00)
#
# ax_ay_df = n_to_bvp_signal([df_AX, df_AY], time, lower_limit=100, upper_limit00)
# ax_az_df = n_to_bvp_signal([df_AX, df_AZ], time, lower_limit=100, upper_limit00)
# az_ay_df = n_to_bvp_signal([df_AZ, df_AY], time, lower_limit=100, upper_limit00)
#
# gx_gy_df = n_to_bvp_signal([df_GX, df_GY], time, lower_limit=100, upper_limit00)
# gx_gz_df = n_to_bvp_signal([df_GX, df_GZ], time, lower_limit=100, upper_limit00)
# gz_gy_df = n_to_bvp_signal([df_GZ, df_GY], time, lower_limit=100, upper_limit00)
#
# gx_gz_gy_df = n_to_bvp_signal([df_GX, df_GZ, df_GY], time, lower_limit=100, upper_limit00)
# az_ay_ax_df = n_to_bvp_signal([df_AZ, df_AY, df_AX], time, lower_limit=100, upper_limit00)
#
# ax_gx_df = n_to_bvp_signal([df_AX, df_GX], time, lower_limit=100, upper_limit=500)
#
# all_df = n_to_bvp_signal([df_AX, df_AY, df_AZ, df_GX, df_GY, df_GZ], time, lower_limit=100, upper_limit=500)
#
#
#
#
# # Convert all dataframes to .csv files to plot them
# ax_df.to_csv("ax.csv")
# ay_df.to_csv("ay.csv")
# az_df.to_csv("az.csv")
# gx_df.to_csv("gx.csv")
# gy_df.to_csv("gy.csv")
# gz_df.to_csv("gz.csv")
#
# ax_ay_df.to_csv("ax_ay.csv")
# ax_az_df.to_csv("ax_az.csv")
# az_ay_df.to_csv("az_ay.csv")
#
# gx_gy_df.to_csv("gx_gy.csv")
# gx_gz_df.to_csv("gx_gz.csv")
# gz_gy_df.to_csv("gz_gy.csv")
#
# ax_gx_df.to_csv("ax_gx.csv")
#
# all_df.to_csv("all_df.csv")
#
# df_AX_overlapped.to_csv("ax_overlapped.csv")
#
#
#
#
# ### DATA EXTRACTION STUFF
# ## Divide different arrays into groups pf (window_sz * frequency)
# ax_array = divide_array("ax.csv", window_sz, hz)
# ay_array = divide_array("ay.csv", window_sz, hz)
# az_array = divide_array("az.csv", window_sz, hz)
# gx_array = divide_array("gx.csv", window_sz, hz)
# gy_array = divide_array("gy.csv", window_sz, hz)
# gz_array = divide_array("gz.csv", window_sz, hz)
#
# ##Extract tensor ax[0] + ay[0] + az[0] + gx[0] + gy[0] + gz[0]
# list_nparrays = array_to_nparray(ax_array, ay_array, az_array, gx_array, gy_array, gz_array, window_sz, hz)
#
#
#
#
# #Plot the csv files
# ax_df.plot(kind='line', x='Time', y='BVP', title='Accelerometer X')
# ay_df.plot(kind='line', x='Time', y='BVP', title='Accelerometer Y')
# az_df.plot(kind='line', x='Time', y='BVP', title='Accelerometer Z')
# gx_df.plot(kind='line', x='Time', y='BVP', title='Gyroscope X')
# gy_df.plot(kind='line', x='Time', y='BVP', title='Gyroscope Y')
# gz_df.plot(kind='line', x='Time', y='BVP', title='Gyroscope Z')
#
#
# ax_ay_df.plot(kind='line', x='Time', y='BVP', title='Accelerometer X & Y')
# ax_az_df.plot(kind='line', x='Time', y='BVP', title='Accelerometer X & Z')
# az_ay_df.plot(kind='line', x='Time', y='BVP', title='Accelerometer Z & Y')
#
# gx_gy_df.plot(kind='line', x='Time', y='BVP', title='Gyroscope X & Y')
# gx_gz_df.plot(kind='line', x='Time', y='BVP', title='Gyroscope X & Z')
# gz_gy_df.plot(kind='line', x='Time', y='BVP', title='Gyroscope Z & Y')
#
#
# az_ay_ax_df.plot(kind='line', x='Time', y='BVP', title='Accelerometer X Y & Z')
# gx_gz_gy_df.plot(kind='line', x='Time', y='BVP', title='Gyroscope X  Y & Z')
#
#
# ax_gx_df.plot(kind='line', x='Time', y='BVP', title='AX & GX')
#
# all_df.plot(kind='line', x='Time', y='BVP', title='Everything')
#
# # final_df.to_csv("final.csv")
# # final_df.plot(kind='line', x= 'Time', y='BVP')
# df_AX_overlapped.plot(kind='line', title='Everything')
#
# plt.show()