import numpy as np
import redpctl as redpctl
import matplotlib.pyplot as plt

rp_c = redpctl.RedCtl()

def x_edge(data, thresh = 0.00): 
    mask1 = (data[:-1] < thresh) & (data[1:] > thresh)
    mask2 = (data[:-1] > thresh) & (data[1:] < thresh)
    rising_edge = np.flatnonzero(mask1)+1
    falling_edge = np.flatnonzero(mask2)+1
    # print(np.flatnonzero(mask1 | mask2)+1)
    return  rising_edge, falling_edge

def get_save(counter = 1, quantity = 800, fname = "square"):
    
    data = rp_c.read(counter = counter, quantity = quantity)
    timeseries = np.array(data)

    rising_edge, falling_edge = x_edge(timeseries[0])
    one_period = rising_edge[1] - rising_edge[0]
    one_periods = []

    for i in timeseries:
        one_periods.append(i[rising_edge[0]:rising_edge[0]+one_period ])

    f=open("dataset/" + fname + "_one.csv",'a')
    np.savetxt(f, (one_periods), fmt='%s' , delimiter=',')
    f.close()

quantity = 800

rp_c.set_gen(wave_form = "square")
get_save(counter = 100, quantity = quantity, fname = "square")
rp_c.set_gen(wave_form = "sine")
get_save(counter = 1, quantity = quantity, fname = "sine")

 
