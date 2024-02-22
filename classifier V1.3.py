import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import CSVLogger

data = np.load('./data/for_classifier/tess_data.npy')
labels = np.load('./data/for_classifier/tess_labels.npy')

sizes = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)

ENC = tf.keras.models.load_model('./models/mae_encoder_speaker_gender_emotion')

Inp = Input(shape=(207, 13))
Ltn = ENC([Inp, Inp])
enc = Model(inputs=Inp, outputs=Ltn)
enc.trainable = False

inp = Input(shape=(207, 13))
ltn = enc(inp)
# ltn = tf.expand_dims(inp, axis=3)
ltn = Conv2D(1, 10, activation="relu", kernel_regularizer=regularizers.l2(0.001), padding="same")(ltn)
ltn = tf.squeeze(ltn)
ltn = LSTM(32, dropout=0.7, activation=ReLU, return_sequences=True)(ltn)
ltn = LSTM(128, dropout=0.7, activation=ReLU, return_sequences=True)(ltn)
ltn = BatchNormalization()(ltn)
ltn = LSTM(512, dropout=0.7, activation=ReLU)(ltn)
ltn = Dense(256, activation=ReLU)(ltn)
out = Dense(200, activation="softmax")(ltn)

for i in range(9):
    _, X_test, _, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_train, _, y_train, _ = train_test_split(data, labels, test_size=1 - sizes[i], random_state=42)

    cls = Model(inputs=inp, outputs=out)
    cls.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss = tf.keras.losses.categorical_crossentropy

    cls.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    log_dir = "./logs/cls/speaker_gender_emotion/" + str(round(100*sizes[i]))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    directory = './logs/cls/speaker_gender_emotion/cls_spk_gen_emo_' + str(round(100*sizes[i])) + '.csv'
    csv_logger = CSVLogger(directory, append=True, separator=',')

    cls.fit(X_train, y_train, shuffle=True, epochs=400, batch_size=10, callbacks=[tensorboard_callback, csv_logger],
            validation_data=(X_test, y_test))
