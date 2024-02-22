import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d

sizee = (0.10, 0.15, 0.2, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
iters = pd.read_csv('./logs/cls/emp/cls_emp_10.csv', skiprows=0)
iters = np.array(iters)
iters = iters[:, 0]

cls_emo = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/emotion/cls_emotion_' + str(round(100 * sizee[i])) + '.csv', skiprows=0)
    tmp = np.array(tmp)
    cls_emo[:, i] = tmp[:399, 3]

cls_emo_gen = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/emotion_gender/cls_emo_gen_' + str(round(100 * sizee[i])) + '.csv', skiprows=0)
    tmp = np.array(tmp)
    cls_emo_gen[:, i] = tmp[:399, 3]

cls_emo_gen_spk = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/emotion_gender_speaker/cls_emo_gen_spk_' + str(round(100 * sizee[i])) + '.csv',
                      skiprows=0)
    tmp = np.array(tmp)
    cls_emo_gen_spk[:, i] = tmp[:399, 3]

cls_emo_spk = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/emotion_speaker/cls_emo_spk_' + str(round(100 * sizee[i])) + '.csv', skiprows=0)
    tmp = np.array(tmp)
    cls_emo_spk[:, i] = tmp[:399, 3]

cls_emo_spk_gen = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/emotion_speaker_gender/cls_emo_spk_gen_' + str(round(100 * sizee[i])) + '.csv',
                      skiprows=0)
    tmp = np.array(tmp)
    cls_emo_spk_gen[:, i] = tmp[:399, 3]

cls_emp = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/emp/cls_emp_' + str(round(100 * sizee[i])) + '.csv',
                      skiprows=0)
    tmp = np.array(tmp)
    cls_emp[:, i] = tmp[:399, 3]

cls_gen = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/gender/cls_gen_' + str(round(100 * sizee[i])) + '.csv', skiprows=0)
    tmp = np.array(tmp)
    cls_gen[:, i] = tmp[:399, 3]

cls_gen_emo = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/gender_emotion/cls_gen_emo_' + str(round(100 * sizee[i])) + '.csv', skiprows=0)
    tmp = np.array(tmp)
    cls_gen_emo[:, i] = tmp[:399, 3]

cls_gen_emo_spk = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/gender_emotion_speaker/cls_gen_emo_spk_' + str(round(100 * sizee[i])) + '.csv',
                      skiprows=0)
    tmp = np.array(tmp)
    cls_gen_emo_spk[:, i] = tmp[:399, 3]

cls_gen_spk = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/gender_speaker/cls_gen_spk_' + str(round(100 * sizee[i])) + '.csv', skiprows=0)
    tmp = np.array(tmp)
    cls_gen_spk[:, i] = tmp[:399, 3]

cls_gen_spk_emo = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/gender_speaker_emotion/cls_gen_spk_emo_' + str(round(100 * sizee[i])) + '.csv',
                      skiprows=0)
    tmp = np.array(tmp)
    cls_gen_spk_emo[:, i] = tmp[:399, 3]

cls_spk = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/speaker/cls_spk_' + str(round(100 * sizee[i])) + '.csv', skiprows=0)
    tmp = np.array(tmp)
    cls_spk[:, i] = tmp[:399, 3]

cls_spk_emo = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/speaker_emotion/cls_spk_emo_' + str(round(100 * sizee[i])) + '.csv', skiprows=0)
    tmp = np.array(tmp)
    cls_spk_emo[:, i] = tmp[:399, 3]

cls_spk_emo_gen = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/speaker_emotion_gender/cls_spk_emo_gen_' + str(round(100 * sizee[i])) + '.csv',
                      skiprows=0)
    tmp = np.array(tmp)
    cls_spk_emo_gen[:, i] = tmp[:399, 3]

cls_spk_gen = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/speaker_gender/cls_spk_gen_' + str(round(100 * sizee[i])) + '.csv', skiprows=0)
    tmp = np.array(tmp)
    cls_spk_gen[:, i] = tmp[:399, 3]

cls_spk_gen_emo = np.zeros([399, 18])
for i in range(18):
    tmp = pd.read_csv('./logs/cls/speaker_gender_emotion/cls_spk_gen_emo_' + str(round(100 * sizee[i])) + '.csv',
                      skiprows=0)
    tmp = np.array(tmp)
    cls_spk_gen_emo[:, i] = tmp[:399, 3]

for i in range(18):
    plt.figure(i)
    #lbl = 'MAE, based on Emotion'
    #plt.plot(iters, gaussian_filter1d(cls_emo[:, i], sigma=100), label=lbl)
    #lbl = 'MAE, based on Emotion and Gender'
    #plt.plot(iters, gaussian_filter1d(cls_emo_gen[:, i], sigma=100), label=lbl)
    lbl = 'MAE, based on Emotion, Gender and Speaker'
    plt.plot(iters, gaussian_filter1d(cls_emo_gen_spk[:, i], sigma=100), label=lbl)
    #lbl = 'MAE, based on Emotion and Speaker'
    #plt.plot(iters, gaussian_filter1d(cls_emo_spk[:, i], sigma=100), label=lbl)
    lbl = 'MAE, based on Emotion, Speaker and Gender'
    plt.plot(iters, gaussian_filter1d(cls_emo_spk_gen[:, i], sigma=100), label=lbl)
    #lbl = 'No MAE'
    #plt.plot(iters, gaussian_filter1d(cls_emp[:, i], sigma=100), label=lbl)
    #lbl = 'MAE, based on Gender'
    #plt.plot(iters, gaussian_filter1d(cls_gen[:, i], sigma=100), label=lbl)
    #lbl = 'MAE, based on Gender and Emotion'
    #plt.plot(iters, gaussian_filter1d(cls_gen_emo[:, i], sigma=100), label=lbl)
    lbl = 'MAE, based on Gender, Emotion and Speaker'
    plt.plot(iters, gaussian_filter1d(cls_gen_emo_spk[:, i], sigma=100), label=lbl)
    #lbl = 'MAE, based on Gender and Speaker'
    #plt.plot(iters, gaussian_filter1d(cls_gen_spk[:, i], sigma=100), label=lbl)
    lbl = 'MAE, based on Gender, Speaker and Emotion'
    plt.plot(iters, gaussian_filter1d(cls_gen_spk_emo[:, i], sigma=100), label=lbl)
    #lbl = 'MAE, based on Speaker'
    #plt.plot(iters, gaussian_filter1d(cls_spk[:, i], sigma=100), label=lbl)
    #lbl = 'MAE, based on Speaker and Emotion'
    #plt.plot(iters, gaussian_filter1d(cls_spk_emo[:, i], sigma=100), label=lbl)
    lbl = 'MAE, based on Speaker, Emotion and Gender'
    plt.plot(iters, gaussian_filter1d(cls_spk_emo_gen[:, i], sigma=100), label=lbl)
    #lbl = 'MAE, based on Speaker and Gender'
    #plt.plot(iters, gaussian_filter1d(cls_spk_gen[:, i], sigma=100), label=lbl)
    lbl = 'MAE, based on Speaker, Gender and Emotion'
    plt.plot(iters, gaussian_filter1d(cls_spk_gen_emo[:, i], sigma=100), label=lbl)
    plt.title('Using ' + str(round(sizee[i] * 100)) + '% of Training Data')
    plt.xlabel('Iteration')
    plt.ylabel('Validation Accuracy')
    plt.legend(loc='best')
plt.show()

