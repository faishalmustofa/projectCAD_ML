import numpy as np
import pywt
import os
import matplotlib.pyplot as plt
import mysql.connector
from database import readData,saveData

# readData, filename = readData()
# data = []
# for i in range(len(readData)):
#     temp = float(readData[i])
#     data.append(temp)
# label = 1
# saveData(data,label,filename)
# print("done")
# filename_hasil = "hasil_bagong.txt"
# label = 0
# id_pasien = 33
# connection = mysql.connector.connect(host='localhost', database='cad_ultrasound', user='root', password='')
# cursor = connection.cursor()
# sql_update = "UPDATE pasien SET hasilproses = '" + filename_hasil + "',label = '"+str(label)+"' WHERE id = "+str(id_pasien)
# cursor.execute(sql_update)
# connection.commit()
#
# if (connection.is_connected()):
#     cursor.close()
#     connection.close()
# print("sukses")
import pickle
# label = [0,0,1,0,1,1,0,1,0,0,0,1,0,1,0,0]
# datamitral = {}

# datamitral = open("C:/xampp/htdocs/projectCAD/public/storage/upload/files/dokter/34_Rizki.txt").readlines()
# datamitral = open("C:/xampp/htdocs/projectCAD/public/storage/upload/files/dokter/47_ika.txt").readlines()

# datamitral = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek1_N.txt").readlines()
# datamitral= open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek2_N.txt").readlines()
# datamitral = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek3_C.txt").readlines()
# datamitral[3] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek4_N.txt").readlines()
# datamitral[4] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek5_C.txt").readlines()
# datamitral[5] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek6_C.txt").readlines()
# datamitral[6] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek7_N.txt").readlines()
# datamitral[7] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek8_C.txt").readlines()
# datamitral[8] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek9_N.txt").readlines()
# datamitral[9] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek10_N.txt").readlines()
# datamitral[10] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek11_N.txt").readlines()
# datamitral[11] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek12_C.txt").readlines()
# datamitral[12] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek13_N.txt").readlines()
# datamitral[13] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek14_C.txt").readlines()
# datamitral[14] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek15_N.txt").readlines()
# datamitral[15] = open("D:\KULIAH\TINGKAT4\TA2\dataset\DATA RS SALAMUN\subjek16_N.txt").readlines()
# print(datamitral[0])
# data = []
# for i in range(len(datamitral)) :
#     temp = float(datamitral[i])
#     data.append(temp)
# print(data)
# print(data)
# data = np.array(data)
# print(data)
# def denoising(signal ):
#     coeffs1 = pywt.wavedec(signal, 'db4', level=6)
#     cA11, cD11, cD21, cD31, cD41, cD51, cD61 = coeffs1
#     HthrescD11 = pywt.threshold(cD11, 0.04, mode='soft')
#     HthrescD21 = pywt.threshold(cD21, 0.04, mode='soft')
#     HthrescD31 = pywt.threshold(cD31, 0.04, mode='soft')
#     HthrescD41 = pywt.threshold(cD41, 0.04, mode='soft')
#     HthrescD51 = pywt.threshold(cD51, 0.04, mode='soft')
#     HthrescD61 = pywt.threshold(cD61, 0.04, mode='soft')
#     Hcoeffs1 = cA11, HthrescD11, HthrescD21, HthrescD31, HthrescD41, HthrescD51, HthrescD61
#     cleancoeffs1 = pywt.waverec(Hcoeffs1, 'db4')
#
#     return cleancoeffs1
# den_data = denoising(data)
# den_data = list(den_data)

# arrayt = [[1,2,3],[1,2,3],[1,2,3]]
# with open("datax.txt", "wb") as internal_filename:
#     pickle.dump(arrayt, internal_filename)

# with open("label_train.txt", "rb") as new_filename:
#     pp = pickle.load(new_filename)
# p = pp[0]
# print(pp)
# plt.title("Signal Phonocardiogram - Subyek Sakit")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.plot(data[:3500])
# plt.show()

x = 1.090675765757
g =  round(x, 3)
print(g)

