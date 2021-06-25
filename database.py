import numpy as np
import mysql.connector
from mysql.connector import Error
import matplotlib.pyplot as plt

def readData():
    connection = mysql.connector.connect(host='localhost',database='cad_ultrasound',user='root',password='')

    sql_select_Query = "SELECT id_pasien,nama,pathdata FROM datasets"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    records = cursor.fetchall()
    data = records[0]
    # nama_pasien = data[1]
    filename = data[2]
    # dataSignal = np.genfromtxt(r"C:/xampp/htdocs/projectCAD/storage/app/public/upload/files/"+filename,delimiter=',')

    ## READ TXT FILE
    dataSignal = []
    my_file = open("C:/xampp/htdocs/projectCAD/public/storage/upload/files/dokter/" + filename, "r")
    for line in my_file.readlines():
        if line[-1:] == "\n":
            dataSignal.append(line[:-1])
        else:
            dataSignal.append(line)
    my_file.close()

    # C:/xampp/htdocs/projectCAD/public/storage/upload/files/hasilproses

    if (connection.is_connected()):
        cursor.close()
        connection.close()
    return dataSignal, filename

def saveData(data,label,filename):
    connection = mysql.connector.connect(host='localhost', database='cad_ultrasound', user='root', password='')
    cursor = connection.cursor()

    filename_hasil = 'hasilproses_'+filename
    with open(r'C:\xampp\htdocs\projectCAD\public\storage\upload/files\hasilproses/' + filename_hasil, 'w') as f:
        for row in data:
            f.write(str(row) + '\n')
    f.close()

    #Select Pasien from database
    sql_select = "SELECT id_pasien,nama,pathdata FROM datasets"
    cursor.execute(sql_select)
    records = cursor.fetchall()
    data = records[0]
    id_pasien = data[0]
    print(label[0])

    sql_update = "UPDATE pasien SET hasilproses = '" + filename_hasil + "',label = '"+str(label[0])+"' WHERE id = "+str(id_pasien)
    cursor.execute(sql_update)
    connection.commit()

    if (connection.is_connected()):
        cursor.close()
        connection.close()

    return print("sukses")

def getFiturEkstraksi():
    connection = mysql.connector.connect(host='localhost',
                                         database='cad_ultrasound',
                                         user='root',
                                         password='')
    cursor = connection.cursor()
    sql_select_Query = "SELECT id_pasien,nama,pathdata FROM datasets"
    cursor.execute(sql_select_Query)
    fiturname = cursor.fetchall()
    fitur = np.genfromtxt(r"C:/xampp/htdocs/projectCAD/storage/app/public/upload/fitur/" + fiturname, delimiter=',')


    if (connection.is_connected()):
        cursor.close()
        connection.close()

    return fitur

def saveFiturEkstraksi(fitur,label):
    connection = mysql.connector.connect(host='localhost',
                                         database='cad_ultrasound',
                                         user='root',
                                         password='')
    cursor = connection.cursor()
    # dbfitur = getFiturEkstraksi()
    # dbfitur.append(fitur)
    fiturname = 'fitur.txt'
    rowfitur = open("C:/xampp/htdocs/projectCAD/public/storage/upload/fitur/"+fiturname, "w")
    for row in range(len(fitur)):
        np.savetxt(rowfitur, row)
    rowfitur.close()

    labelname = 'label.txt'
    rowlabel = open("C:/xampp/htdocs/projectCAD/public/storage/upload/fitur/"+labelname, "w")
    for row in range(len(label)):
        np.savetxt(rowlabel,row)
    rowlabel.close()

    sql_update = "UPDATE fitur_ekstraksis SET fitur = '" + fiturname + "', label = '" + labelname + "' WHERE id = 1"
    cursor.execute(sql_update)
    connection.commit()

    if (connection.is_connected()):
        cursor.close()
        connection.close()

    return print("sukses")