import glob
import librosa
import os
import numpy as np
from python_speech_features import mfcc
from scipy.signal import resample

statement_1 = np.empty([1, 360, 207, 13])
root = 'C:\\Datasets\\My RAVDESS\\To Remove Gender\\Statement 1\\'
for i in range(2):
    path = root + str(i+1)
    samples = np.empty([1, 207, 13])
    for filename in glob.glob(os.path.join(path, '*.wav')):
        (sig, rate) = librosa.load(filename, sr=None)
        sig = resample(sig, num=100000)
        c = mfcc(sig, rate, nfft=2048)
        c = np.expand_dims(c, axis=0)
        samples = np.concatenate((samples, c), axis=0)
    samples = samples[1:, :, :]
    samples = np.expand_dims(samples, axis=0)
    statement_1 = np.concatenate((statement_1, samples), axis=0)
statement_1 = statement_1[1:, :, :, :]
print(statement_1.shape)

statement_2 = np.empty([1, 360, 207, 13])
root = 'C:\\Datasets\\My RAVDESS\\To Remove Gender\\Statement 2\\'
for i in range(2):
    path = root + str(i+1)
    samples = np.empty([1, 207, 13])
    for filename in glob.glob(os.path.join(path, '*.wav')):
        (sig, rate) = librosa.load(filename, sr=None)
        sig = resample(sig, num=100000)
        c = mfcc(sig, rate, nfft=2048)
        c = np.expand_dims(c, axis=0)
        samples = np.concatenate((samples, c), axis=0)
    samples = samples[1:, :, :]
    samples = np.expand_dims(samples, axis=0)
    statement_2 = np.concatenate((statement_2, samples), axis=0)
statement_2 = statement_2[1:, :, :, :]
print(statement_2.shape)

rav = np.empty([1, 2, 207, 13])

inp_1 = statement_1[0, :, :, :]
inp_2 = statement_1[1, :, :, :]

for i in range(60):
    for j in range(60):
        tmp_1 = inp_1[i, :, :]
        tmp_2 = inp_2[j, :, :]
        tmp_1 = np.expand_dims(tmp_1, axis=0)
        tmp_2 = np.expand_dims(tmp_2, axis=0)
        tmp = np.concatenate((tmp_1, tmp_2), axis=0)
        tmp_1 = np.expand_dims(tmp_1, axis=0)
        tmp = np.expand_dims(tmp, axis=0)
        rav = np.concatenate((rav, tmp), axis=0)

del statement_1

inp_1 = statement_2[0, :, :, :]
inp_2 = statement_2[1, :, :, :]

for i in range(60):
    for j in range(60):
        tmp_1 = inp_1[i, :, :]
        tmp_2 = inp_2[j, :, :]
        tmp_1 = np.expand_dims(tmp_1, axis=0)
        tmp_2 = np.expand_dims(tmp_2, axis=0)
        tmp = np.concatenate((tmp_1, tmp_2), axis=0)
        tmp_1 = np.expand_dims(tmp_1, axis=0)
        tmp = np.expand_dims(tmp, axis=0)
        rav = np.concatenate((rav, tmp), axis=0)

del statement_2

print(rav.shape)
np.save('./data/for_mae/mae_gender.npy', rav)
