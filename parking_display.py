import time
import serial
import string
import datetime

try:
    uart = serial.Serial(
        port='/dev/ttyS0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
)
except:
    print("An exception occurred")
    
#pre definitions
capacity = 50
free = 0
used = capacity - free


#%test_string = "Ahoj".encode('utf-8')
#cislo = '5000' .encode ('utf-8')7
#bytes_sent = uart.write("{}".format(free).encode("utf-8"))
#print("Sended", bytes_sent, "byte")
#recieve = uart.read(bytes_sent)
#print ("Received ",len(recieve))
#print(recieve)

#display four digits
display = ("{}")
full = ("FULL" .encode("utf-8"))
#print(display.format(free, used))
if free > 0:
    send = uart.write(display.format(free) .encode("utf-8"))
    recieve = uart.read(send)
    print(recieve)
    
elif free <= 0:
    send = uart.write(full)
    recieve = uart.read(send)
    print(recieve)
  
  
#dim on time
    
def diming(i): #0 stands for display off, 1 stands for dimmest, 4 stands for not dim.default as 4
    dim ="atf2=({})"
    send = uart.write(dim.format(i) .encode("utf-8"))
    recieve = uart.read(send)
    print(recieve)
    
dim = False
now = datetime.datetime.now()

while True:
    now = datetime.datetime.now()
    if((now.hour == 18) and (dim == False)):
        diming(3)
        dim = True
    if((dim == True) and (now.hour == 6)) :
        diming(4)
        dim = False
     
        
    
    

         