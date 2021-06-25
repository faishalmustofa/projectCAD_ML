import paho.mqtt.client as mqttClient
import time, csv, os
import sys

# Masukan data ke txt
def inputToFileTxt(fileName,data):
    if (os.path.isfile(fileName+'.txt')):
        #Appending
        f = open(fileName+'.txt', 'a')
        f = open("C:/xampp/htdocs/projectCAD/public/storage/upload/files/" + fileName + ".txt", "a")
        f.write(data+' \n')
        #print(value)
    else:
        #Create data txt
        f = open(fileName+'.txt', 'w')
        f = open("C:/xampp/htdocs/projectCAD/public/storage/upload/files/" + fileName + ".txt", "w")
        f.write(data+' \n')
        #print(value)
    f.close()

#Split data
def inputToFile(value,nameFileData,nameFileWaktu):
    ArrValuex = value.split("|")
    if (len(ArrValuex) == 2):
        inputToFileTxt(nameFileWaktu,ArrValuex[0])
        inputToFileTxt(nameFileData,ArrValuex[1])
    else:
        print("Data not valid : "+value)

# Check file dengan nomor tertentu sudah ada atau belum
def Check_file(nameFileData):
    fileNotFound = True
    countFile = 1
    while(fileNotFound):
        if (os.path.isfile(nameFileData+str(countFile)+'.txt')):
            countFile = countFile+1
        else:
            fileNotFound = False
    return countFile

def on_connect(client, userdata, flags, rc):
    if rc == 0:

        print("Connected to broker")

        global Connected  # Use global variable
        Connected = True  # Signal connection

    else:

        print("Connection failed")


# Check Data yang dikirim sudah selesai atau belum
def on_message(client, userdata, msg):
    nameFileData = "datasignal"
    nameFileWaktu = "waktusignal"
    a = msg.payload.decode("utf-8")
    if (a != "END" or a != "end") :
        inputToFile(a,nameFileData,nameFileWaktu)
        # print(a)
    else:
        number = Check_file(nameFileData)
        os.rename("datasignal.txt", "datasignal"+str(number)+".txt")
        os.rename("waktusignal.txt", "waktusignal"+str(number)+".txt")
        sys.exit('END DATA')


#Edit this with your environment
Connected = False   #global variable for the state of the connection

broker_address= str(sys.argv[1])  #Broker address
port = 1883                         #Broker port
topic = str(sys.argv[2])

print("IP Broker\t",broker_address,":",port)
print("topic\t\t"+topic)
print("CTRL + C to convert to txt")
print("")

client = mqttClient.Client()               #create new instance
client.on_connect= on_connect                      #attach function to callback
client.on_message= on_message                      #attach function to callback
client.connect(broker_address,port,60) #connect
client.subscribe(topic) #subscribe
try:
    client.loop_forever()
except Exception:
    print("ERROR Try again")
    os.remove("datasignal.txt")

# Running program
# python3 sub_mqtt.py ip_broker topik
# contohnya :  python broker.py 192.168.43.90 datacad
