import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import CSVLogger

data = np.load('./data/for_classifier/tess_data.npy')
labels = np.load('./data/for_classifier/tess_labels.npy')

# sizee = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
sizee = (0.40, 0.50, 0.60, 0.70, 0.80, 0.90)

ENC = tf.keras.models.load_model('./models/mae_encoder_gender_speaker_emotion')

Inp = Input(shape=(207, 13))
Ltn = ENC([Inp, Inp])
enc = Model(inputs=Inp, outputs=Ltn)
enc.trainable = False

inp = Input(shape=(207, 13))
ltn = enc(inp)
# ltn = tf.expand_dims(inp, axis=3)
ltn = Conv2D(8, 20, activation="relu", kernel_regularizer=regularizers.l2(0.001), padding="same")(ltn)
ltn = Dropout(0.2)(ltn)
ltn = Conv2D(16, 20, activation="relu", kernel_regularizer=regularizers.l2(0.001), padding="same")(ltn)
ltn = Dropout(0.2)(ltn)
ltn = Conv2D(32, 20, activation="relu", kernel_regularizer=regularizers.l2(0.001), padding="same")(ltn)
ltn = Dropout(0.2)(ltn)
ltn = BatchNormalization()(ltn)
ltn = GlobalAveragePooling2D()(ltn)
ltn = Dense(16, activation="relu")(ltn)
out = Dense(200, activation="softmax")(ltn)

for i in range(6):
    _, X_test, _, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    X_train, _, y_train, _ = train_test_split(data, labels, test_size=1 - sizee[i], random_state=42)

    cls = Model(inputs=inp, outputs=out)
    cls.summary()
    # plot_model(cls, to_file='./figures/cls_mae_both.png', show_shapes=True, show_layer_names=True)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    loss = tf.keras.losses.categorical_crossentropy

    cls.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    log_dir = "./logs/cls/gender_emotion_speaker/" + str(round(100*sizee[i]))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    drct = './logs/cls/gender_emotion_speaker/cls_gen_emo_spk_' + str(round(100*sizee[i])) + '.csv'
    csv_logger = CSVLogger(drct, append=True, separator=',')

    cls.fit(X_train, y_train, shuffle=True, epochs=400, batch_size=10, callbacks=[tensorboard_callback, csv_logger],
            validation_data=(X_test, y_test))
