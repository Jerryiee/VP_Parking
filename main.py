import functions_display
import parking

while True:
    now = datetime.datetime.now()
    if((now.hour == 18) and (dim == False)):
        diming(3)
        dim = True
    elif((dim == True) and (now.hour == 6)) :
        diming(4)
        dim = False
