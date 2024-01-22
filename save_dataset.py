import numpy as np
import redpctl as redpctl
import matplotlib.pyplot as plt
# from time import sleep


rp_c = redpctl.RedCtl()

rp_c.set_gen(wave_form = "square")

nrows = 100
data_square = rp_c.read(counter = nrows)
timeseries_square = np.array(data_square)

fs1=open("square.csv",'a')
np.savetxt(fs1, (timeseries_square), fmt='%s' , delimiter=',')
fs1.close()

rp_c.set_gen(wave_form = "sine")

nrows = 2
data_sine = rp_c.read(counter = nrows)
timeseries_sine = np.array(data_sine)

fs2=open("sine.csv",'a')
np.savetxt(fs2, (timeseries_sine), fmt='%s' , delimiter=',')
fs2.close()
 
