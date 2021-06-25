import pickle
from math import *
import numpy as np
import pywt
import scipy.stats
import matlab.engine
import matplotlib.pyplot as plt
from database import readData,saveData

def denoising(signal):
    coeffs1 = pywt.wavedec(signal, 'db10', level=4)
    cA11, cD11, cD21, cD31, cD41 = coeffs1
    HthrescD11 = pywt.threshold(cD11, 0.04, mode='hard')
    HthrescD21 = pywt.threshold(cD21, 0.04, mode='hard')
    HthrescD31 = pywt.threshold(cD31, 0.04, mode='hard')
    HthrescD41 = pywt.threshold(cD41, 0.04, mode='hard')
    Hcoeffs1 = cA11, HthrescD11, HthrescD21, HthrescD31, HthrescD41
    cleancoeffs1 = pywt.waverec(Hcoeffs1, 'db10')
    data_denoising = list(cleancoeffs1)

    return data_denoising

def createData(data):
    hasil = []
    for i in range(len(data)):
        temp_i = data[i]
        temp = []
        for j in range(len(temp_i)):
            temp_j = temp_i[j][0]
            temp.append(temp_j)
        hasil.append(temp)
    return hasil

def shannon_energy(coef):

    energy = []
    for i in range(len(coef)) :
        temp = coef[i]
        for l in range(len(temp[0:])):
            sum_energy = sum([-((p**2)*log2(p**2)) for p in temp if p > 0])
            energy.append(sum_energy)

    return energy

def shannon_entropy(coef):

    entropy = []
    for i in range(len(coef)):
        temp = coef[i]
        for l in range(len(temp[0:])) :
            sum_entropy = sum([-p*log2(p) for p in temp if p > 0])
            entropy.append(sum_entropy)
    return entropy

def time_feature(data):
    temp = []
    for i in range(len(data)):
        temp = temp + data[i]
    mean,std = calculate_statistics(temp)
    skew = scipy.stats.skew(temp)
    kurt = scipy.stats.kurtosis(temp)

    return mean, std, skew, kurt

def calculate_statistics(list_values):
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)

    return mean, std

def EkstraksiFitur(S1,Systole,S2,Diastole):

    ## Fitur Entropy
    entropy_S1 = shannon_entropy(S1)
    entropy_Systole = shannon_entropy(Systole)
    entropy_S2 = shannon_entropy(S2)
    entropy_Diastole = shannon_entropy(Diastole)

    mEntropy_S1, sdEntropy_S1 = calculate_statistics(entropy_S1)
    mEntropy_Systole, sdEntropy_Systole = calculate_statistics(entropy_Systole)
    mEntropy_S2, sdEntropy_S2 = calculate_statistics(entropy_S2)
    mEntropy_Diastole, sdEntropy_Diastole = calculate_statistics(entropy_Diastole)

    ## Fitur Energy
    energy_S1 = shannon_energy(S1)
    energy_Systole = shannon_energy(Systole)
    energy_S2 = shannon_energy(S2)
    energy_Diastole = shannon_energy(Diastole)

    mEnergy_S1, sdEnergy_S1 = calculate_statistics(energy_S1)
    mEnergy_Systole, sdEnergy_Systole = calculate_statistics(energy_Systole)
    mEnergy_S2, sdEnergy_S2 = calculate_statistics(energy_S2)
    mEnergy_Diastole, sdEnergy_Diastole = calculate_statistics(energy_Diastole)

    ##Fitur Time Domain
    mTD_S1, sdTD_S1,skew_S1,kurt_S1 = time_feature(S1)
    mTD_Systole, sdTD_Systole,skew_Systole,kurt_Systole = time_feature(Systole)
    mTD_S2, sdTD_S2,skew_S2, kurt_S2 = time_feature(S2)
    mTD_Diastole, sdTD_Diastole, skew_Diastole, kurt_Diastole = time_feature(Diastole)

    featureEnergy_Entropy_TimeDomain = [
        mTD_S1, sdTD_S1, skew_S1, kurt_S1,
        mEnergy_S1, sdEnergy_S1, mEntropy_S1, sdEntropy_S1,
        mTD_Systole, sdTD_Systole, skew_Systole, kurt_Systole,
        mEnergy_Systole, sdEnergy_Systole, mEntropy_Systole, sdEntropy_Systole,
        mTD_S2, sdTD_S2, skew_S2, kurt_S2,
        mEnergy_S2, sdEnergy_S2, mEntropy_S2, sdEntropy_S2,
        mTD_Diastole, sdTD_Diastole, skew_Diastole, kurt_Diastole,
        mEnergy_Diastole, sdEnergy_Diastole, mEntropy_Diastole, sdEntropy_Diastole
    ]
    features = featureEnergy_Entropy_TimeDomain
    return features

label_train = [0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,1,1,1,1]
sample_rate = 200

# datamitral = open("C:/xampp/htdocs/projectCAD/public/storage/upload/files/datasignal.txt").readlines()
#
# data = []
# for i in range(len(datamitral)) :
#     temp = float(datamitral[i])
#     data.append(temp)

data = []
# xdata,filename = readData()
xdata = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek1_N.txt").readlines()
for i in range(len(xdata)):
    temp = float(xdata[i])
    data.append(temp)

## Denoising
denois_data = denoising(data)

## Segmentasi Data
print("Segmentasi Start")
eng = matlab.engine.start_matlab()
signal = matlab.double(denois_data)
[PCG_resampled, S1, Systole, S2, Diastole] = eng.challenge(signal, sample_rate, nargout=5)
data_S1 = createData(S1)
data_Systole = createData(Systole)
data_S2 = createData(S2)
data_Diastole = createData(Diastole)

siklus_jantung = data_Systole[0]
# print(data_Systole)
# with open('sinyal Systole.txt', 'w') as f:
#     for item in data_Systole:
#         f.write("%s\n" % item)
# with open('sinyal S2.txt', 'w') as f:
#     for item in data_S2:
#         f.write("%s\n" % item)
# with open('sinyal Diastole.txt', 'w') as f:
#     for item in data_Diastole:
#         f.write("%s\n" % item)
# print("done")

# print(siklus_jantung)

plt.plot(siklus_jantung[:265])
plt.xlabel("Time(m/s)")
plt.ylabel("Amplitudo(A)")
plt.show()


# print("Ekstraksi Fitur")
# # Feature Extraction
# all_features_test = []
# features = EkstraksiFitur(data_S1,data_Systole,data_S2,data_Diastole)
# all_features_test.append(features)
# # print(all_features_test)
#
# with open("all_feature_Energy_Entropy_TimeDomain.txt", "rb") as new_filename:
#     xfeatures = pickle.load(new_filename)
# all_features_train = xfeatures
#
# print("Klasifikasi")
# #Klasifikasi
# from sklearn.naive_bayes import GaussianNB
# data_train_NB = np.array(all_features_train)
# label_train_NB = np.array(label_train)
# acc_NB_Gaussian = []
# rata2_sens = []
# rata2_spec = []
# data_test_NB = np.array(all_features_test)
# label_test_NB = [0]
#
# X_train, X_test, y_train, y_test = data_train_NB, data_test_NB, label_train_NB,label_test_NB
#
# classifier = GaussianNB()
# classifier.fit(X_train, y_train)
# predict = classifier.predict(X_test)
# print()
#
# if (predict == 0):
#     print("Subyek Sehat")
# else:
#     print("Subyek Sakit")
#
# saveData(denois_data,predict,filename)