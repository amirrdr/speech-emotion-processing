import numpy as np
import tensorflow as tf
from keras.layers import *
from keras import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

data = np.load('./data/for_classifier/savee_data.npy')
labels = np.load('./data/for_classifier/savee_labels.npy')
labels = np.argmax(labels, axis=1)

data = data[labels == 0]

ENC = tf.keras.models.load_model('./models/mae_encoder_emotion')

Inp = Input(shape=(207, 13))
Ltn = ENC([Inp, Inp])
enc = Model(inputs=Inp, outputs=Ltn)
enc.trainable = False

data = enc(data)
data = data[:, :, :, 0]
data = np.mean(data, axis=2)
# plt.pcolormesh(data, norm=LogNorm(), cmap='inferno')
# plt.colorbar()
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_data, y_data = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

x_data = x_data.flatten()
y_data = y_data.flatten()
z_data = data.flatten()
# z_data = z_data / np.linalg.norm(z_data)
ax.bar3d(x_data, y_data, np.zeros(len(z_data)), 1, 1, z_data)

plt.show()
