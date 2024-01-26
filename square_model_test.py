import numpy as np
import pandas as pd
import keras
from matplotlib import pyplot as plt


model = keras.models.load_model("model_square.keras")
dot_img_file = 'model_square.png'
keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

df_square = pd.read_csv("dataset/square.csv", header=None)
df_sine = pd.read_csv("dataset/sine.csv", header=None)
# df_square = pd.read_csv("dataset/square_one.csv", header=None)
# df_sine = pd.read_csv("dataset/sine_one.csv", header=None)


df_square = df_square.transpose()
df_sine = df_sine.transpose()

training_mean = df_square.mean()
training_std = df_square.std()
df_training_value = (df_square - training_mean) / training_std
print("Number of training samples:", len(df_training_value))

TIME_STEPS = 248

# Generated training sequences for use in sthe model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    # print(len([item for sublist in output for item in sublist]))
    return np.stack(output)

x_train = create_sequences(df_training_value.values)

# x_train = df_training_value[0].to_list()
# x_train = x_train[:TIME_STEPS]
# x_train = np.array(x_train)
# x_train = df_training_value[0].to_numpy()
# x_train = np.reshape(x_train, (1,)+x_train.shape)
# x_train=x_train[:,np.newaxis]
# x_train = x_train[..., None]
# x_train = np.expand_dims(x_train, axis=1)
print("Training input shape: ", x_train.shape)

x_train_pred = model.predict(x_train)

print(len(x_train_pred))
for i in x_train_pred:
    plt.plot(i)

plt.show()

exit()

train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)

df_merged = pd.concat([df_square,df_sine],ignore_index=True, sort=False)
df_test_value = (df_merged - training_mean) / training_std

x_test = create_sequences(df_test_value.values)
print("Test input shape: ", x_test.shape)

x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

anomalous_data_indices = []
for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)

df_subset = df_merged.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df_merged.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r")
plt.show()