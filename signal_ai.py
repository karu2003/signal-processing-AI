import numpy as np
import matplotlib.pyplot as plt
import redpctl as redpctl
import keras
import librosa
import librosa.display
from sklearn.preprocessing import normalize

import tensorflow as tf
np.random.seed(42)

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h = array.shape[0]
    w = array.shape[1]
    a = max((xx - h) // 2,0)
    aa = max(0,xx - a - h)
    b = max(0,(yy - w) // 2)
    bb = max(yy - b - w,0)
    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')

def generate_features(y_cut):
    max_size = 1000 #my max audio file feature width
    n_fft = 255
    sr = 1000000
    hop_length = 512 
    stft = padding(np.abs(librosa.stft(y_cut, n_fft=n_fft, hop_length = hop_length)), 128, max_size)
    MFCCs = padding(librosa.feature.mfcc(y_cut, n_fft=n_fft, hop_length=hop_length,n_mfcc=128),128,max_size)
    spec_centroid = librosa.feature.spectral_centroid(y=y_cut, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y_cut, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y_cut, sr=sr)
    #Now the padding part
    image = np.array([padding(normalize(spec_bw),1, max_size)]).reshape(1,max_size)
    image = np.append(image,padding(normalize(spec_centroid),1, max_size), axis=0) 
    #repeat the padded spec_bw,spec_centroid and chroma stft until they are stft and MFCC-sized
    for i in range(0,9):
        image = np.append(image,padding(normalize(spec_bw),1, max_size), axis=0)
        image = np.append(image, padding(normalize(spec_centroid),1, max_size), axis=0)
        image = np.append(image, padding(normalize(chroma_stft),12, max_size), axis=0)
    image=np.dstack((image,np.abs(stft)))
    image=np.dstack((image,MFCCs))
    return image


rp_c = redpctl.RedCtl()
nrows = 1
epochs = 200
data = rp_c.read(counter = nrows)

timeseries = np.array(data)

# min_val = np.min(timeseries)
# max_val = np.max(timeseries)
# timeseries = (timeseries - min_val) / (max_val - min_val)

# layer = keras.layers.LayerNormalization()
# timeseries = layer(timeseries)

for i in timeseries:
    # plt.plot(i)
    spectrogram = tf.signal.stft(i, frame_length=int(len(i)/2), frame_step=1)
    spectrogram = tf.abs(spectrogram)
    # spectrogram = spectrogram[..., tf.newaxis]
    plt.plot(spectrogram)
    # plt.imshow(spectrogram)
    plt.show()
    # print(spectrogram)

exit()
# plt.show()

# Y = np.zeros(nrows)

Y = np.ones(nrows)


TrainSet = np.random.choice(timeseries.shape[0], int(timeseries.shape[0]*0.80), replace=False)
XTrain = timeseries[TrainSet,:]
YTrain = Y[TrainSet]
TestSet = np.delete(np.arange(len(Y)), TrainSet)
XTest = timeseries[TestSet,:]
YTest = Y[TestSet]
sampleWeight = np.ones_like(YTrain)
sampleWeight[YTrain==0] = 0.3

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
# model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
dot_img_file = 'model_1.png'
keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
# keras.utils.plot_model(model, show_shapes=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# history = model.fit(XTrain, YTrain, epochs=100, verbose=True, sample_weight=sampleWeight)
history = model.fit(XTrain, YTrain, epochs=epochs, verbose=True)
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

