import numpy as np
import redpctl as redpctl
import matplotlib.pyplot as plt

def x_edge(data, thresh = 0.00): 
    mask1 = (data[:-1] < thresh) & (data[1:] > thresh)
    mask2 = (data[:-1] > thresh) & (data[1:] < thresh)
    rising_edge = np.flatnonzero(mask1)+1
    falling_edge = np.flatnonzero(mask2)+1
    return  rising_edge, falling_edge

rp_c = redpctl.RedCtl()

rp_c.set_gen(wave_form = "square")

nrows = 1
quantity = 2000

data_square = rp_c.read(counter = nrows, quantity = quantity)
timeseries_square = np.array(data_square)

rising_edge, falling_edge = x_edge(timeseries_square[0])
x_periods = []

for i in timeseries_square:
    x_periods.append(i[rising_edge[0]:rising_edge[-1]])

fs1=open("dataset/square.csv",'a')
np.savetxt(fs1, (x_periods), fmt='%s' , delimiter=',')
fs1.close()

rp_c.set_gen(wave_form = "sine")

nrows = 1
data_sine = rp_c.read(counter = nrows, quantity = quantity)
timeseries_sine = np.array(data_sine)

rising_edge, falling_edge = x_edge(timeseries_sine[0])
x_periods = []

for i in timeseries_sine:
    x_periods.append(i[rising_edge[0]:rising_edge[-1]])

fs2=open("dataset/sine.csv",'a')
np.savetxt(fs2, (x_periods), fmt='%s' , delimiter=',')
fs2.close()
 
