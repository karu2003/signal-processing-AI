import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

timeseries_square = np.genfromtxt("square.csv", delimiter=',')
# timeseries_square = np.loadtxt("square.csv",delimiter=',')
# df1 = pd.read_csv(r"square.csv",header=None) 
# timeseries_square = np.array(df1)


for i in timeseries_square:
    plt.plot(i)

plt.show()

timeseries_sine = np.genfromtxt("sine.csv", delimiter=',')
# timeseries_sine = np.loadtxt("sine.csv",delimiter=',',skiprows=0)
# df2 = pd.read_csv(r"sine.csv",header=None) 
# timeseries_sine = np.array(df2)

for i in timeseries_sine:
    plt.plot(i)

plt.show()
