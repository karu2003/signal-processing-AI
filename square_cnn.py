import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from sklearn.preprocessing import normalize, minmax_scale
from sklearn import preprocessing, model_selection
import tensorflow as tf

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 2

epochs = 100

timeseries = np.genfromtxt("square.csv", delimiter=',')
nrows = len(timeseries)
Y = np.ones(nrows)

yTrue = np.random.choice(nrows, int(nrows*0.2), replace=False)
Y[yTrue] = 0

# timeseries_mean = np.mean(timeseries)
t_max = np.max(timeseries)
t_min = np.min(timeseries)
print(t_max,t_min)

for i in yTrue:
    data = np.random.normal(0, 1, size=len(timeseries[i]))
    normalizedData = minmax_scale(data,feature_range=(t_min,t_max))                
    timeseries[i] = normalizedData


# for i in yTrue:
#     plt.plot(timeseries[i])
# plt.show()

# for i in timeseries:
#     plt.plot(i)
# plt.show()  

# scaler = preprocessing.MinMaxScaler()
# series_list = [ scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in timeseries]
# num_classes = 1
# labels_list = Y

# XTrain, XTest, YTrain, YTest = model_selection.train_test_split(
#     series_list, labels_list, test_size=0.2, random_state=42, shuffle=True
# )

# train_dataset = tf.data.Dataset.from_tensor_slices((XTrain, YTrain))
# test_dataset = tf.data.Dataset.from_tensor_slices((XTest, YTest))

# train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
# test_dataset = test_dataset.batch(BATCH_SIZE)

# timeseries = np.array(series_list)

# samples = np.random.normal(0, 1, size=nrows)

TrainSet = np.random.choice(timeseries.shape[0], int(timeseries.shape[0]*0.80), replace=False)
XTrain = timeseries[TrainSet,:]
YTrain = Y[TrainSet]
TestSet = np.delete(np.arange(len(Y)), TrainSet)
XTest = timeseries[TestSet,:]
YTest = Y[TestSet]
sampleWeight = np.ones_like(YTrain)
sampleWeight[YTrain==1] = 0.3

# exit()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout 
from tensorflow.keras.layers import Conv1D,MaxPooling1D
from tensorflow.compat.v1.random import set_random_seed
set_random_seed(42)

model = Sequential()
model.add(Conv1D(400, kernel_size=10, activation='relu', use_bias=False,
                input_shape=(timeseries.shape[1],1)))
# model.add(MaxPooling1D(pool_size=4))
# model.add(Conv1D(6, kernel_size=8, activation='relu', use_bias=False))
# model.add(MaxPooling1D(pool_size=4))
# model.add(Conv1D(1, kernel_size=4, activation='relu', use_bias=False))
# model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
dot_img_file = 'model_1.png'
keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
# keras.utils.plot_model(model, show_shapes=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(XTrain, YTrain, epochs=100, verbose=True, sample_weight=sampleWeight)
# history = model.fit(XTrain, YTrain, epochs=epochs, verbose=True)
model.summary()
print(model.evaluate(XTest, YTest, verbose=False))
plt.figure()
plt.plot(history.history['loss'], color='k', label='loss',marker='*')
plt.plot(history.history['accuracy'], color='r', label='accuracy',marker='+')
plt.legend()

yP = model.predict(XTest)
print(yP)
plt.show()

model.save("model_500.keras") # save_format='tf' save_format='h5'

# Now, we can simply load without worrying about our custom objects.
reconstructed_model = keras.models.load_model("model_500.keras")

# Let's check:
np.testing.assert_allclose(
    model.predict(XTrain), reconstructed_model.predict(XTrain)
)

