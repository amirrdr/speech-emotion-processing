import numpy as np
import tensorflow as tf
from keras.layers import *
from keras import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = np.load('./data/for_classifier/savee_data.npy')
labels = np.load('./data/for_classifier/savee_labels.npy')
labels = np.argmax(labels, axis=1)

data, _, labels, _ = train_test_split(data, labels, test_size=0.5, random_state=42)

ENC = tf.keras.models.load_model('./models/mae_encoder_emotion')

Inp = Input(shape=(207, 13))
Ltn = ENC([Inp, Inp])
enc = Model(inputs=Inp, outputs=Ltn)
enc.trainable = False

# set up the figure and axes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# data_comp = enc(data)
# data_comp = np.mean(data, axis=2)
n_fake_data = np.max(labels)

for i in range(2):
    _x = data[labels == i]
    _y = labels[labels == i]
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    top = x + y
    bottom = np.zeros_like(top)
    width = depth = 1
    ax.bar3d(x, y, bottom, width, depth, top, shade=True)

plt.show() 
