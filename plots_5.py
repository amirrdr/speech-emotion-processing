import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d

sizee = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
iters = pd.read_csv('./logs/cls/emp/cls_emp_10.csv', skiprows=0)
iters = np.array(iters)
iters = iters[1:, 0]


cls_gen_emo_spk = np.zeros([399, 18])
for i in range(9):
    tmp = pd.read_csv('./logs/cls/gender_emotion_speaker/cls_gen_emo_spk_' + str(round(100 * sizee[i])) + '.csv',
                      skiprows=0)
    tmp = np.array(tmp)
    cls_gen_emo_spk[:, i] = tmp[:399, 3]

cls_spk_gen_emo = np.zeros([399, 18])
for i in range(9):
    tmp = pd.read_csv('./logs/cls/speaker_gender_emotion/cls_spk_gen_emo_' + str(round(100 * sizee[i])) + '.csv',
                      skiprows=0)
    tmp = np.array(tmp)
    cls_spk_gen_emo[:, i] = tmp[:399, 3]

cls_spk_emo_gen = np.zeros([399, 18])
for i in range(9):
    tmp = pd.read_csv('./logs/cls/speaker_emotion_gender/cls_spk_emo_gen_' + str(round(100 * sizee[i])) + '.csv',
                      skiprows=0)
    tmp = np.array(tmp)
    cls_spk_emo_gen[:, i] = tmp[:399, 3]

cls_emp = np.zeros([399, 18])
for i in range(9):
    tmp = pd.read_csv('./logs/cls/emp/cls_emp_' + str(round(100 * sizee[i])) + '.csv', skiprows=0)
    tmp = np.array(tmp)
    cls_emp[:, i] = tmp[:399, 3]

plts = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sigma = 100

for i in range(len(plts)):
    plt.subplot(3, 3, i + 1)
    lbl = 'Speaker, Emotion and Gender'
    plt.plot(iters, gaussian_filter1d(cls_spk_emo_gen[:, plts[i]], sigma=sigma), label=lbl)
    lbl = 'Gender, Emotion and Speaker'
    plt.plot(iters, gaussian_filter1d(cls_gen_emo_spk[:, plts[i]], sigma=sigma), label=lbl)
    lbl = 'Speaker, Gender and Emotion'
    plt.plot(iters, gaussian_filter1d(cls_spk_gen_emo[:, plts[i]], sigma=sigma), label=lbl)
    lbl = 'No MAE'
    plt.plot(iters, gaussian_filter1d(cls_emp[:, plts[i]], sigma=sigma), label=lbl)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy, Using ' + str(round(sizee[plts[i]] * 100)) + '% of Data')
    plt.legend()
plt.show()
