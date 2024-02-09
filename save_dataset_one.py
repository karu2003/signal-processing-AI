import numpy as np
import redpctl as redpctl
import matplotlib.pyplot as plt

thresh = 0.2
rp_c = redpctl.RedCtl(dec=1, trig = thresh)

def get_save(counter = 1, quantity = 800, fname = "square"):
    
    data = rp_c.read(counter = counter, quantity = quantity)
    timeseries = np.array(data)

    rising_edge, falling_edge = rp_c.x_edge(timeseries[0], thresh = thresh)
    one_period = rising_edge[1] - rising_edge[0]
    one_periods = []

    for i in timeseries:
        one_periods.append(i[rising_edge[0]:rising_edge[0]+one_period ])

    f=open("dataset/" + fname + "_one.csv",'a')
    np.savetxt(f, (one_periods), fmt='%s' , delimiter=',')
    f.close()

rx_buffer_size = 16384

# rp_c.set_gen(wave_form = "square")
rp_c.set_burst(wave_form="sine")
get_save(counter = 50, quantity = rx_buffer_size, fname = "square")
exit()
rp_c.set_gen(wave_form = "sine")
get_save(counter = 1, quantity = quantity, fname = "sine")

 
