import numpy as np
import matplotlib.pyplot as plt
import redpctl as redpctl
import keras
np.random.seed(42)


rp_c = redpctl.RedCtl()
nrows = 100
data = rp_c.read(counter = nrows)

timeseries = np.array(data)

# for i in timeseries:
#     plt.plot(i)
# plt.show()

Y = np.zeros(nrows)


TrainSet = np.random.choice(timeseries.shape[0], int(timeseries.shape[0]*0.80), replace=False)
XTrain = timeseries[TrainSet,:]
YTrain = Y[TrainSet]
TestSet = np.delete(np.arange(len(Y)), TrainSet)
XTest = timeseries[TestSet,:]
YTest = Y[TestSet]
sampleWeight = np.ones_like(YTrain)
sampleWeight[YTrain==0] = 1.0 # 0.3

# exit()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout 
from tensorflow.keras.layers import Conv1D,MaxPooling1D
from tensorflow.compat.v1.random import set_random_seed
set_random_seed(42)

model = Sequential()
model.add(Conv1D(8, kernel_size=10, activation='relu', use_bias=False,
input_shape=(timeseries.shape[1],1)))
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(6, kernel_size=8, activation='relu', use_bias=False))
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(1, kernel_size=4, activation='relu', use_bias=False))
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(XTrain, YTrain, epochs=100, verbose=True, sample_weight=sampleWeight)
history = model.fit(XTrain, YTrain, epochs=100, verbose=True)
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

