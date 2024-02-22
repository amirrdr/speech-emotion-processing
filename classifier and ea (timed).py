from glob import glob
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras import Model
from sklearn.model_selection import train_test_split
from keras import regularizers
from time import time
import pandas as pd

data = np.load('./data/for_classifier/savee_data.npy')
labels = np.load('./data/for_classifier/savee_labels.npy')
en_names = glob('./models/*/', recursive = True)
print('Models:')
print(en_names)

for i in range(len(en_names)):
    rti = 10
    tms = []
    while rti < 96:
        _, X_test, _, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        X_train, _, y_train, _ = train_test_split(data, labels, test_size=1-(rti/100), random_state=42)

        ENC = tf.keras.models.load_model(en_names[i])
        ENC.summary()

        Inp = Input(shape=(207, 13))
        Ltn = ENC([Inp, Inp])
        enc = Model(inputs=Inp, outputs=Ltn)
        enc.trainable = False

        inp = Input(shape=(207, 13))
        ltn = enc(inp)
        ltn = Conv2D(8, 20, activation="relu", kernel_regularizer=regularizers.l2(0.001), padding="same")(ltn)
        ltn = Dropout(0.2)(ltn)
        ltn = Conv2D(16, 20, activation="relu", kernel_regularizer=regularizers.l2(0.001), padding="same")(ltn)
        ltn = Dropout(0.2)(ltn)
        ltn = Conv2D(32, 20, activation="relu", kernel_regularizer=regularizers.l2(0.001), padding="same")(ltn)
        ltn = Dropout(0.2)(ltn)
        ltn = BatchNormalization()(ltn)
        ltn = GlobalAveragePooling2D()(ltn)
        ltn = Dense(16, activation="relu")(ltn)
        out = Dense(15, activation="softmax")(ltn)

        cls = Model(inputs=inp, outputs=out)
        cls.summary()

        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
        loss = tf.keras.losses.categorical_crossentropy

        cls.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

        start = time()
        cls.fit(X_train, y_train, shuffle=True, epochs=400, batch_size=10, validation_data=(X_test, y_test))
        trvl_time = time()-start
        start = time()
        cls.evaluate(X_test, y_test)
        evl_time = time()-start
        tms.append((trvl_time, evl_time))

        rti += 5
    tms_save = pd.DataFrame(tms)
    tms_save.to_excel(en_names[i] + '/times.xlsx')