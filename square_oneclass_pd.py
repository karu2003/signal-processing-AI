import numpy as np
from sklearn import svm
import pandas as pd
from matplotlib import pyplot as plt

df_square_one = pd.read_csv("dataset/square_one.csv", header=None)
df_square_one = df_square_one.transpose()

# print(df_square_one.head())

training_mean = df_square_one.mean()
training_std = df_square_one.std()
df_training_value = (df_square_one - training_mean) / training_std
# print("Number of training samples:", len(df_training_value))


def create_sequences(df):
    output = []
    for i in range(len(df.columns)):
        output.append(df[i].values)
    return np.stack(output)


# X_train = df_training_value.to_numpy()
# X_train = df_training_value.iloc[:,-1:].values
X_train = create_sequences(df_training_value)

for i in X_train:
    plt.plot(i)
plt.show()

print("Training input shape square: ", X_train.shape)
y = np.linspace(0, X_train.shape[0], X_train.shape[0], dtype=np.int16)
# y = np.ones(X_train.shape[0])

# clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf = svm.SVC()
clf.fit(X_train, y)

# y_pred_train = clf.predict(X_train)
# n_error_train = y_pred_train[y_pred_train == -1].size
# print("Error_train ",n_error_train)


def test_probe(file="sine", clf=clf):
    df_probe = pd.read_csv("dataset/" + file + ".csv", header=None)
    df_probe = df_probe.transpose()
    # print(df_probe.head())
    probe_mean = df_probe.mean()
    probe_std = df_probe.std()
    df_probe_value = (df_probe - probe_mean) / probe_std
    # print("Number of sine samples:", len(df_probe_value))

    X_probe = df_probe_value[0].values[:250]
    print("Probe input shape: ", X_probe.shape)

    # y_pred_probe = clf.predict([X_probe])
    y_pred_probe = clf.predict(X_probe.reshape(1, -1))
    n_error_train = y_pred_probe[y_pred_probe == -1].size
    # print(y_pred_probe)
    print(file + " error_train ", n_error_train)

    fig, ax = plt.subplots()
    plt.plot(X_probe)
    plt.show()


test_probe(file="sine", clf=clf)
test_probe(file="square", clf=clf)
