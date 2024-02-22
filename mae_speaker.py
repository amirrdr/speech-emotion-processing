import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import CSVLogger

data = np.load('./data/for_mae/mae_speaker_5.npy')
data = data[1:, :, :, :]
data = np.transpose(data, [1, 0, 2, 3])
input_1 = data[0, :, :, :]
input_2 = data[1, :, :, :]
input_1 = np.squeeze(input_1)
input_2 = np.squeeze(input_2)
print(input_1.shape)
print(input_2.shape)

inp_1 = Input(shape=(207, 13))
inp_2 = Input(shape=(207, 13))

ltn_1 = tf.expand_dims(inp_1, axis=3)
ltn_1 = Conv2D(3, 10, padding="same", activation="relu")(ltn_1)
ltn_1 = Dropout(0.2)(ltn_1)
ltn_1 = Conv2D(3, 10, padding="same", activation="relu")(ltn_1)
ltn_1 = Dropout(0.2)(ltn_1)
ltn_1 = Conv2D(3, 10, padding="same", activation="relu")(ltn_1)
ltn_1 = Dropout(0.2)(ltn_1)

ltn_2 = tf.expand_dims(inp_2, axis=3)
ltn_2 = Conv2D(3, 10, padding="same", activation="relu")(ltn_2)
ltn_2 = Dropout(0.2)(ltn_2)
ltn_2 = Conv2D(3, 10, padding="same", activation="relu")(ltn_2)
ltn_2 = Dropout(0.2)(ltn_2)
ltn_2 = Conv2D(3, 10, padding="same", activation="relu")(ltn_2)
ltn_2 = Dropout(0.2)(ltn_2)

merged = tf.concat((ltn_1, ltn_2), axis=3)

merged = Conv2D(3, 10, padding="same", activation="relu")(merged)
merged = Dropout(0.2)(merged)

encoded = Conv2D(1, 10, padding="same")(merged)
encoded = Dropout(0.2)(encoded)

mae_encoder = Model(inputs=[inp_1, inp_2], outputs=encoded, name="mae_encoder")
mae_encoder.summary()

ltn_1 = Conv2D(3, 10, padding="same", activation="relu")(encoded)
ltn_1 = Dropout(0.2)(ltn_1)
ltn_1 = Conv2D(3, 10, padding="same", activation="relu")(ltn_1)
ltn_1 = Dropout(0.2)(ltn_1)
ltn_1 = Conv2D(3, 10, padding="same", activation="relu")(ltn_1)
ltn_1 = Dropout(0.2)(ltn_1)
ltn_1 = Conv2D(1, 10, padding="same")(ltn_1)
ltn_1 = Dropout(0.2)(ltn_1)
out_1 = tf.squeeze(ltn_1)


ltn_2 = Conv2D(3, 10, padding="same", activation="relu")(encoded)
ltn_2 = Dropout(0.2)(ltn_2)
ltn_2 = Conv2D(3, 10, padding="same", activation="relu")(ltn_2)
ltn_2 = Dropout(0.2)(ltn_2)
ltn_2 = Conv2D(3, 10, padding="same", activation="relu")(ltn_2)
ltn_2 = Dropout(0.2)(ltn_2)
ltn_2 = Conv2D(1, 10, padding="same")(ltn_2)
ltn_2 = Dropout(0.2)(ltn_2)
out_2 = tf.squeeze(ltn_2)

mae = Model(inputs=[inp_1, inp_2], outputs=[out_1, out_2], name="mae")
mae.summary()
plot_model(mae, to_file='./figures/mae.png', show_shapes=True, show_layer_names=True)

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)


def my_custom_loss(y_pred, y_true):
    return K.mean(K.square(y_pred - y_true), axis=-1)/2


mae.compile(optimizer=opt, loss=my_custom_loss)

time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = "./logs/mae_speaker/" + time
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
csv_logger = CSVLogger('./logs/mae_speaker.csv', append=True, separator=',')

mae.fit([input_1, input_2], [input_1, input_2], shuffle=True, epochs=150, batch_size=64, callbacks=[
    tensorboard_callback, csv_logger], validation_split=0.2)
mae.evaluate([input_1, input_2], [input_1, input_2])

mae_encoder.trainable = False
mae_encoder.save('./models/mae_encoder_speaker')
