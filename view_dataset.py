import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

timeseries_square = np.genfromtxt("square.csv", delimiter=',')
# timeseries_square = np.loadtxt("square.csv",delimiter=',',skiprows=0)
# df = pd.read_csv(r"square.csv") 
# timeseries_square = np.array(df)

for i in timeseries_square:
    plt.plot(i)

timeseries_sine = np.genfromtxt("sine.csv", delimiter=',')
# timeseries_sine = np.loadtxt("sine.csv",delimiter=',',skiprows=0)
# df = pd.read_csv(r"sine.csv") 
# timeseries_sine = np.array(df)

print(len(timeseries_square))     
print(len(timeseries_sine))
for i in timeseries_sine:
    plt.plot(i)
plt.show()
 
