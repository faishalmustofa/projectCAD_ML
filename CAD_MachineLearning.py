import pickle
from math import *
import numpy as np
import pywt
import scipy.stats
import matlab.engine
from database import readData,saveData
label = [0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0,1,1,1,1]
datamitral = {}
datamitral[0] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek1_N.txt").readlines()
datamitral[1] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek2_N.txt").readlines()
datamitral[2] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek3_C.txt").readlines()
datamitral[3] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek4_N.txt").readlines()
datamitral[4] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek5_C.txt").readlines()
datamitral[5] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek6_C.txt").readlines()
datamitral[6] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek7_N.txt").readlines()
datamitral[7] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek8_C.txt").readlines()
datamitral[8] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek9_N.txt").readlines()
datamitral[9] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek10_N.txt").readlines()
datamitral[10] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek11_N.txt").readlines()
datamitral[11] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek12_C.txt").readlines()
datamitral[12] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek13_N.txt").readlines()
datamitral[13] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek14_C.txt").readlines()
datamitral[14] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek15_N.txt").readlines()
datamitral[15] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek16_N.txt").readlines()
datamitral[16] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek17_C.txt").readlines()
datamitral[17] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek18_C.txt").readlines()
datamitral[18] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek19_C.txt").readlines()
datamitral[19] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek20_C.txt").readlines()

xdata = {}
for j in range(len(datamitral)) :
    temp_array = []
    for i in range(len(datamitral[j])) :
        x = datamitral[j]
        temp = float(x[i])
        temp_array.append(temp)
    xdata[j] = temp_array
sample_rate = 200

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

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

    snr = signaltonoise(data_denoising)

    return data_denoising, snr

def hitung_datasegmen(data):
    hasil = 0
    for i in range(len(data)):
        temp = len(data[i])
        hasil = hasil+temp
    return hasil

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
    # mean_feature.append(mean)
    # std_feature.append(std)
    rkurt = round(kurt, 3)
    rskew = round(skew, 2)

    return mean, std, rskew, rkurt

def calculate_statistics(list_values):
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)

    rmean = round(mean, 3)
    rstd = round(std, 3)

    return rmean, rstd

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

    featureEntropy = [
        mEntropy_S1,mEntropy_Systole,mEntropy_S2,mEntropy_Diastole,
        sdEntropy_S1,sdEntropy_Systole,sdEntropy_S2,sdEntropy_Diastole
    ]
    featureEnergy = [
        mEnergy_S1,mEnergy_Systole,mEnergy_S2,mEnergy_Diastole,
        sdEnergy_S1,sdEnergy_Systole,sdEnergy_S2,sdEnergy_Diastole
    ]

    featureTimeDomain = [
        mTD_S1, sdTD_S1,skew_S1,kurt_S1,
        mTD_Systole,sdTD_Systole,skew_Systole,kurt_Systole,
        mTD_S2, sdTD_S2,skew_S2,kurt_S2,
        mTD_Diastole, sdTD_Diastole,skew_Diastole,kurt_Diastole
    ]

    featureEnergy_Entropy = [
        mEnergy_S1, mEnergy_Systole, mEnergy_S2, mEnergy_Diastole,
        sdEnergy_S1, sdEnergy_Systole, sdEnergy_S2, sdEnergy_Diastole,
        mEntropy_S1,mEntropy_Systole,mEntropy_S2,mEntropy_Diastole,
        sdEntropy_S1,sdEntropy_Systole,sdEntropy_S2,sdEntropy_Diastole
    ]

    featureEnergy_TimeDomain = [
        mEnergy_S1, mEnergy_Systole, mEnergy_S2, mEnergy_Diastole,
        sdEnergy_S1, sdEnergy_Systole, sdEnergy_S2, sdEnergy_Diastole,
        mTD_S1, sdTD_S1, skew_S1, kurt_S1,
        mTD_Systole, sdTD_Systole, skew_Systole, kurt_Systole,
        mTD_S2, sdTD_S2, skew_S2, kurt_S2,
        mTD_Diastole, sdTD_Diastole, skew_Diastole, kurt_Diastole
    ]

    featureEntropy_TimeDomain = [
        mEntropy_S1, mEntropy_Systole, mEntropy_S2, mEntropy_Diastole,
        sdEntropy_S1, sdEntropy_Systole, sdEntropy_S2, sdEntropy_Diastole,
        mTD_S1, sdTD_S1, skew_S1, kurt_S1,
        mTD_Systole, sdTD_Systole, skew_Systole, kurt_Systole,
        mTD_S2, sdTD_S2, skew_S2, kurt_S2,
        mTD_Diastole, sdTD_Diastole, skew_Diastole, kurt_Diastole
    ]

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
    features = featureTimeDomain
    return features

print("===================== MEMULAI SISTEM =====================")

## Denoising
data = {}
snr = []
for idx in range(len(xdata)) :
    data[idx],temp_snr = denoising(xdata[idx])
    data_snr = round(float(temp_snr),3)
    snr.append(data_snr)

print("Nilai SNR dataset : ")
print(snr)
with open('SNR_Denoising.txt', 'w') as f:
    # f.write("[mean S1, std S1, mean Systole, std Systole, mean S2, std S2, mean Diastole, std Diastole]")
    for item in snr:
        f.write("%s\n" % item)
eng = matlab.engine.start_matlab()

print("Segmentasi Data & Ekstraksi Fitur")
all_features_test = []
energy = []
entropy = []
timedomain = []
energy_entropy = []
energy_timedomain = []
all_S1_sehat = 0
all_S1_sakit = 0
all_sys_sehat = 0
all_sys_sakit = 0
all_S2_sehat = 0
all_S2_sakit = 0
all_dia_sehat = 0
all_dia_sakit = 0
for i in range(len(data)) :
    signal = matlab.double(data[i])
    [PCG_resampled, S1, Systole, S2, Diastole] = eng.challenge(signal, sample_rate, nargout=5)
    data_S1 = createData(S1)
    data_Systole = createData(Systole)
    data_S2 = createData(S2)
    data_Diastole = createData(Diastole)
    if (label[i] == 0) :
        all_S1_sehat = all_S1_sehat + hitung_datasegmen(data_S1)
        all_sys_sehat = all_sys_sehat + hitung_datasegmen(data_Systole)
        all_S2_sehat = all_S2_sehat + hitung_datasegmen(data_S2)
        all_dia_sehat = all_dia_sehat + hitung_datasegmen(data_Diastole)
    else:
        all_S1_sakit = all_S1_sehat + hitung_datasegmen(data_S1)
        all_sys_sakit = all_sys_sehat + hitung_datasegmen(data_Systole)
        all_S2_sakit = all_S2_sehat + hitung_datasegmen(data_S2)
        all_dia_sakit = all_dia_sehat + hitung_datasegmen(data_Diastole)

    # Feature Extraction
    features = EkstraksiFitur(data_S1,data_Systole,data_S2,data_Diastole)

    all_features_test.append(features)
    energy.append(features_Energy)
    entropy.append(features_Entropy)
    timedomain.append(features_TimeDomain)
    energy_entropy.append(features_Energy_Entropy)
    energy_timedomain.append(features_Energy_TimeDomain)
    print(i+1)

print("Jumlah data segmen S1 subyek sehat :", all_S1_sehat)
print("Jumlah data segmen S1 subyek sakit :", all_S1_sakit)
print("Jumlah data segmen Systole subyek sehat :", all_sys_sehat)
print("Jumlah data segmen Systole subyek sakit :", all_sys_sakit)
print("Jumlah data segmen S2 subyek sehat :", all_S2_sehat)
print("Jumlah data segmen S2 subyek sakit :", all_S2_sakit)
print("Jumlah data segmen Diastole subyek sehat :", all_dia_sehat)
print("Jumlah data segmen Diastole subyek sakit:", all_dia_sakit)


print("Jumlah Data : ",len(all_features_test))
print(all_features_test)
with open('feature_TimeDomain.txt', 'w') as f:
    f.write("[mean S1, std S1, mean Systole, std Systole, mean S2, std S2, mean Diastole, std Diastole]")
    for item in all_features_test:
        f.write("%s\n" % item)

## WRITE & READ FILE HASIL EKSTRAKSI FITUR

with open("all_feature_Energy_Entropy_TimeDomain.txt", "wb") as internal_filename:
    pickle.dump(all_features_test, internal_filename)
# with open("all_feature_Entropy.txt", "wb") as internal_filename:
#     pickle.dump(entropy, internal_filename)
# with open("all_feature_TimeDomain.txt", "wb") as internal_filename:
#     pickle.dump(timedomain, internal_filename)
# with open("all_feature_Energy_Entropy.txt", "wb") as internal_filename:
#     pickle.dump(energy_entropy, internal_filename)
# with open("all_feature_Energy_TimeDomain.txt", "wb") as internal_filename:
#     pickle.dump(energy_timedomain, internal_filename)
# with open("all_feature_Energy.txt", "rb") as new_filename:
#     xdata = pickle.load(new_filename)
# with open("all_feature_Energy_Entropy_TimeDomain.txt", "rb") as new_filename:
#     xdata = pickle.load(new_filename)
print("DONE")
# all_features_test = xdata
print("Fitur Ekstraksi : ",all_features_test)
print("Jumlah Data : ", len(all_features_test))

print('===================== KLASIFIKASI  ======================')
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
print('Gaussian Naive Bayes')
# data_train_NB = np.array(all_features_train)
# label_train_NB = np.array(label_train)
# print(label_train_NB)
acc_NB_Gaussian = []
rata2_sens = []
rata2_spec = []
data_test_NB = np.array(all_features_test)
label_test_NB = np.array(label)
# print(data_NB)
# print("Label Data : ",label_NB)
kf = KFold(n_splits = 5, shuffle=False)
i=1
for train_idx, test_idx in kf.split(data_test_NB):
    X_train, X_test, y_train, y_test = data_test_NB[train_idx], data_test_NB[test_idx], label_test_NB[train_idx], label_test_NB[test_idx]

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    predict = classifier.predict(X_test)
    # print("Predict : ",predict)
    accuracy = accuracy_score(y_test, predict)
    print('Akurasi NB Gaussian ke ', i, ' : ', accuracy * 100, "%")
    from sklearn.metrics import classification_report
    print(classification_report(y_test, predict))
    # print(confusion_matrix(y_test, predict))
    acc_NB_Gaussian.insert(i, accuracy)
    i += 1

    cm = confusion_matrix(y_test, predict)
    print(cm)
    TN = cm[1][1] * 1.0
    FN = cm[1][0] * 1.0
    TP = cm[0][0] * 1.0
    FP = cm[0][1] * 1.0
    total = TN + FN + TP + FP
    print('TP', TP, 'TN', TN, 'FP', FP, 'FN', FN)
    sens = TN / (TN + FP) * 100
    spec = TP / (TP + FN) * 100
    rata2_sens.insert(i,sens)
    rata2_spec.insert(i,spec)
    print('Sensitivity : ' + str(sens))
    print('Specificity : ' + str(spec))
print()

print(" Kesimpulan dari sistem yaitu Rata-rata Akurasi 5-Fold Cross Validation")
print()
print("Rata-rata akurasi dengan Fitur Energy, Entropy dan Time Domain : ", np.mean(acc_NB_Gaussian) * 100, "%")
print("Rata-rata Sensitivity dengan Fitur Energy, Entropy dan Time Domain : ", np.mean(rata2_sens), "%")
print("Rata-rata Specificity dengan Fitur Energy, Entropy dan Time Domain : ", np.mean(rata2_spec), "%")