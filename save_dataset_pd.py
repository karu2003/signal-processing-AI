import numpy as np
import redpctl as redpctl
import matplotlib.pyplot as plt
# from time import sleep
import pandas as pd

rp_c = redpctl.RedCtl()

rp_c.set_gen(wave_form = "square")

nrows = 10
data_square = rp_c.read(counter = nrows)
timeseries_square = np.array(data_square)
df = pd.DataFrame(timeseries_square)
# df = pd.DataFrame(np.matrix.transpose(timeseries_square))

fs1=open("square_pd.csv",'a')
df.to_csv(fs1, index = False, header = False)
fs1.close()

rp_c.set_gen(wave_form = "sine")

nrows = 2
data_sine = rp_c.read(counter = nrows)
timeseries_sine = np.array(data_sine)

fs2=open("sine_pd.csv",'a')
np.savetxt(fs2, (timeseries_sine), fmt='%s' , delimiter=',')
fs2.close()
 
