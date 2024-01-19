import numpy as np
import matplotlib.pyplot as plt
import redpctl as redpctl
import keras


rp_c = redpctl.RedCtl()
nrows = 1
data = rp_c.read(counter = nrows)

timeseries = np.array(data)

model = keras.models.load_model("model_500.keras")
seq_predictions=model.predict(timeseries)
print(seq_predictions)

for i in timeseries:
    plt.plot(i)
plt.show()



# print(seq_predictions >= 0.5).astype(int)

# print('Outputs shape')    
# print(seq_predictions.shape)
# seq_predictions=np.transpose(seq_predictions)[0]
# print(seq_predictions.shape)
# seq_predictions = list(map(lambda x: 0 if x<0.5 else 1, seq_predictions))