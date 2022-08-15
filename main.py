
# importing  all the
# functions defined in functions_display.py
from functions_display import *
import datetime
from gpiozero import Button
from time import sleep

button1 = Button(2)
button2 = Button(3)
spaces_displayed = 0

dim = False #flag dim on time
full_flag = False

blinking(0)
freq_blink(6)
with open('spaces.txt','r') as f:
    free_spaces = int(f.read())
    spaces_displayed = free_spaces
send_serial(free_spaces)

now = datetime.datetime.now()


while True:


    now = datetime.datetime.now()
    if now.hour == 18 and dim == False:
        diming(2)
        dim = True
    elif dim == True and now.hour == 6:
        diming(4)
        dim = False
    """decrease or increase parking slots
    manualy button1 decrease, button2 increase"""


    if button1.is_pressed: 
        print("Decrease")
        Decrease()
        sleep(0.25)
    elif button2.is_pressed:
        print("Increase")
        Increase()
        sleep(0.25)
    """ spec if you push button, you may
    calibrate manualy free_spaces"""


    with open('spaces.txt','r') as f:
        free_spaces = int(f.read())

        if free_spaces <= 0 and full_flag == False :
            full()
            full_flag = True

        elif free_spaces > 0 and full_flag == True:
            full_flag = False
            blinking(0)
            send_serial(free_spaces)
        
        elif free_spaces > 0 and free_spaces != spaces_displayed:
            send_serial(free_spaces)
            spaces_displayed = free_spaces