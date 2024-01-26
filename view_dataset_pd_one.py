import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
df_square = pd.read_csv("dataset/square_one.csv", header=None)
df_sine = pd.read_csv("dataset/sine_one.csv", header=None)

df_square = df_square.transpose()
df_sine = df_sine.transpose()

print(df_square.head())
print(df_sine.head())

fig, ax = plt.subplots()
df_square.plot(legend=False, ax=ax)

fig, ax = plt.subplots()
df_sine.plot(legend=False, ax=ax)
plt.show()