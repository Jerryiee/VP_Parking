
# importing  all the
# functions defined in functions_display.py
from functions_display import *
import datetime
from gpiozero import Button
from time import sleep

button1 = Button(2)
button2 = Button(3)

#display one digit
display = ("{}")
full = ("FULL" .encode("utf-8"))

#dim on time
dim = False #flag
now = datetime.datetime.now()

while True:
    now = datetime.datetime.now()
    if((now.hour == 18) and (dim == False)):
        diming(3)
        dim = True
    elif((dim == True) and (now.hour == 6)) :
        diming(4)
        dim = False
    """decrease or increase parking slots
    manualy button1 decrease, button2 increase"""
    if button1.is_pressed: 
        print("Decrease")
        Decrease()
        sleep(0.5)
    elif button2.is_pressed:
        print("Increase")
        Increase()
        sleep(0.5)