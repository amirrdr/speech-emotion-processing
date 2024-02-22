import matplotlib.pyplot as plt
from matplotlib import cm
import librosa
import tensorflow as tf
from keras.layers import *
from keras import Model
import numpy as np
from python_speech_features import mfcc
from scipy.signal import resample
from librosa.feature.inverse import mfcc_to_audio
import soundfile as sf

SR = 5000
idx = 1
NM = 13

en_name = './models/mae_encoder_emotion_and_gender/'
root = './data/test/f (' + str(idx) + ').WAV'
(sig_, rate) = librosa.load(root, sr=None)
sig = resample(sig_, num=100000)
c = mfcc(sig, rate, nfft=2048)
c_ = mfcc(sig_, rate)

print(c.shape)
fig, ax = plt.subplots()
mfcc_data= np.swapaxes(c, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.turbo, origin='lower')
ax.set_title('MFCC of Original Sample', fontsize='xx-large')

plt.show()

aud = mfcc_to_audio(c_, sr=rate,)
c_rec = mfcc(aud, rate, nfft=2048)

print(c_rec.shape)
fig, ax = plt.subplots()
mfcc_data= np.swapaxes(c_rec, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.turbo, origin='lower')
ax.set_title('MFCC of Reconstracted Input', fontsize='xx-large')

plt.show()

ENC = tf.keras.models.load_model(en_name)

Inp = Input(shape=(207, 13))
Ltn = ENC([Inp, Inp])
enc = Model(inputs=Inp, outputs=Ltn)
enc.trainable = False

out = enc(np.expand_dims(c, axis=0))
out = np.array(out, dtype='float')

print(out[0, :, :, 0].shape)

fig, ax = plt.subplots()
mfcc_data= np.swapaxes(out[0, :, :, 0], 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.turbo, origin='lower')
ax.set_title('Output of Encoder', fontsize='xx-large')

plt.show()

aud_ = mfcc_to_audio(out[0, :, :, 0], sr=SR, n_fft=2048, n_mels=NM)
aud_ = np.array(aud_, dtype='float')

sf.write('./data/test/reconstructed.wav', aud, rate)
sf.write('./data/test/features.wav', aud_, SR, 'PCM_24')

print(len(aud))
print(len(aud_))
