from time import sleep
import serial

uart = serial.Serial(
        port='/dev/ttyS0', #Replace ttyS0 with ttyAM0 for Pi1,Pi2,Pi0
        baudrate = 115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
)
# After each string or command sent, a 5ms delay or wait for display to return an āEā is needed

def clear_display():
    reset = "atd0=()" .encode("utf-8")
    uart.write(reset)
    delay_uart()
    
def blinking(i): # 1 start blinking, 0-stop blinking
    turn = "ate0=({})"
    uart.write(turn.format(i) .encode("utf-8"))
    delay_uart()
    
def freq_blink(i): #1-10 - default as 5
    freq = "ate1=({})"
    uart.write(freq.format(i) .encode("utf-8"))
    delay_uart()

def diming(i): #0 stands for display off, 1 stands for dimmest, 4 stands for not dim.default as 4
    dim ="atf2=({})"
    send = uart.write(dim.format(i) .encode("utf-8"))
    recieve = uart.read(send)
    print(recieve)
    delay_uart()
    
def send_serial(i):
    s = "  {}"
    uart.write(s.format(i) .encode("utf-8"))
    delay_uart()
    
def delay_uart(): #After each string or command sent, a 5ms delay is needed
    sleep(0.05)

def full():
    full = ("FULL" .encode("utf-8"))
    uart.write(full)
    delay_uart()
    blinking(1)
    delay_uart()

#save to file and read previous value of free spaces

def Increase():
    
    with open('spaces.txt','r') as f:
        free_spaces = int(f.read())
        free_spaces += 1

    with open('spaces.txt','w') as f2:
        f2.truncate() # clear previous content
        f2.write(f'{str(free_spaces)}')

def Decrease():
    
    with open('spaces.txt','r') as f:
        free_spaces = int(f.read())
        free_spaces -= 1

    with open('spaces.txt','w') as f2:
        f2.truncate() # clear previous content
        f2.write(f'{str(free_spaces)}')
