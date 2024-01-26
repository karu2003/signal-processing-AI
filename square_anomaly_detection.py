import numpy as np
import pandas as pd
import keras
from keras import layers
from matplotlib import pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
df_square_one = pd.read_csv("dataset/square_one.csv", header=None)
df_square = pd.read_csv("dataset/square.csv", header=None)
df_sine = pd.read_csv("dataset/sine.csv", header=None)

df_square_one = df_square_one.transpose()
df_square = df_square.transpose()
df_sine = df_sine.transpose()

# df_square_join = pd.Series(df_square_one.values.ravel('F'))

# print(df_square.head())
# print(df_square_one.head())
# print(df_sine.head())

# fig, ax = plt.subplots()
# df_square.plot(legend=False, ax=ax)
# fig, ax = plt.subplots()
# df_sine.plot(legend=False, ax=ax)
# fig, ax = plt.subplots()
# df_square_join.plot(legend=False, ax=ax)
# plt.show()


# Normalize and save the mean and std we get,
# for normalizing test data.
training_mean = df_square.mean()
training_std = df_square.std()
df_training_value = (df_square - training_mean) / training_std
print("Number of training samples:", len(df_training_value))

# fig, ax = plt.subplots()
# df_training_value.plot(legend=False, ax=ax)
# plt.show()

TIME_STEPS = 248

# Generated training sequences for use in sthe model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

x_train = create_sequences(df_training_value.values)
print("Training input shape: ", x_train.shape)

"""
## Build a model
"""


model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Conv1DTranspose(
            filters=16,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32,
            kernel_size=7,
            padding="same",
            strides=2,
            activation="relu",
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

"""
## Train the model
"""

history = model.fit(
    x_train,
    x_train,
    epochs=100,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)

"""
Let's plot training and validation loss to see how the training went.
"""

# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()
# plt.show()


# Get train MAE loss.
x_train_pred = model.predict(x_train)

train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

# plt.hist(train_mae_loss, bins=50)
# plt.xlabel("Train MAE loss")
# plt.ylabel("No of samples")
# plt.show()

# Get reconstruction loss threshold.
threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)

"""
### Compare recontruction

Just for fun, let's see how our model has recontructed the first sample.
This is the 288 timesteps from day 1 of our training dataset.
"""

# Checking how the first sequence is learnt
# plt.plot(x_train[0])
# plt.plot(x_train_pred[0])
# plt.show()

"""
### Prepare test data
"""

df_merged = pd.concat([df_square,df_sine],ignore_index=True, sort=False)

# df_test_value = (df_sine - training_mean) / training_std
df_test_value = (df_merged - training_mean) / training_std
# df_test_value = (df_square - training_mean) / training_std

# fig, ax = plt.subplots()
# df_test_value.plot(legend=False, ax=ax)
# plt.show()

# Create sequences from test values.
x_test = create_sequences(df_test_value.values)
print("Test input shape: ", x_test.shape)

# Get test MAE loss.
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

# plt.hist(test_mae_loss, bins=50)
# plt.xlabel("test MAE loss")
# plt.ylabel("No of samples")
# plt.show()

# Detect all the samples which are anomalies.
anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

# data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
anomalous_data_indices = []
for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)

"""
Let's overlay the anomalies on the original test data plot.
"""

# df_subset = df_sine.iloc[np.where(anomalies)]

df_subset = df_merged.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
df_merged.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r")
plt.show()

model.save("model_square.keras")

df_training_value = (df_square - training_mean) / training_std
x_train = create_sequences(df_training_value.values)
print("Training input shape: ", x_train.shape)

# Now, we can simply load without worrying about our custom objects.
reconstructed_model = keras.models.load_model("model_square.keras")

# Let's check:
np.testing.assert_allclose(
    model.predict(x_train), reconstructed_model.predict(x_train)
)