import numpy as np
from sklearn import svm
import pandas as pd
from matplotlib import pyplot as plt


def normalize(v: list) -> list:
    return v / np.linalg.norm(v)


timeseries = np.genfromtxt("dataset/square_one.csv", delimiter=",")

training_mean = timeseries.mean()
training_std = timeseries.std()
timeseries_porm = (timeseries - training_mean) / training_std

# for i in timeseries_porm:
#     plt.plot(i)
# plt.show()

X = []
y = []
for ixd, i in enumerate(timeseries_porm):
    X.append(timeseries_porm[ixd].tolist())
    y.append(ixd + 1)

samples_svm = timeseries_porm.shape[1]

clf = svm.SVC()

clf.fit(X, y)
print(clf.predict(timeseries_porm[0].reshape(1, -1)))


def test_predict(file="square"):
    test_form = np.genfromtxt("dataset/" + file + ".csv", delimiter=",")
    test_norm = (test_form - training_mean) / training_std
    # print("Number of test samples:", len(test_norm))

    # X = []
    # for i in range(0, test_norm.shape[0], samples_svm):
    #     X.append(test_norm[i:i+samples_svm-1].tolist())

    # XX = np.reshape(test_norm, (int(test_norm.size / samples_svm), samples_svm))
    test_norm.shape = (test_norm.size//samples_svm, samples_svm)

    for i in test_norm:
        plt.plot(i)
    plt.show()

    # return clf.predict(XX)
    return clf.predict(test_norm)


print(test_predict(file="square"))
print(test_predict(file="sine"))
