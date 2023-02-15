from tkinter import *

def get_text(root, val, name):
    try:
        with open(name, "r") as f:
            lines = f.readlines()
            nums = lines[1].split()
            val.set(nums[1])
    except IOError as e:
        print(e)
    else:
        root.after(1000, lambda: get_text(root, val, name))

def get_text2(root, val, name):
    try:
        with open(name, "r") as f:
            lines = f.readlines()
            nums = lines[1].split()
            val.set(nums[2])
    except IOError as e:
        print(e)
    else:
        root.after(1000, lambda: get_text2(root, val, name))

def quit_win():
    root.destroy()

root = Tk()
root.minsize(1080,1920)
#root.attributes('-fullscreen', True)
root.configure(bg='white')

eins = StringVar()
eins2 = StringVar()

data12 = Label(root, text="volne:", bg='white', fg= "green")
data12.config(font=('Jersey M54 Font',100)) 
data12.pack()

data1 = Label(root, textvariable=eins, bg='white', fg= "green")
data2 = Label(root, textvariable=eins2, bg='white', fg= "red")

data1.config(font=('Jersey M54 Font',200)) 
data1.pack(pady=20)

data13 = Label(root, text="obsadene:", bg='white', fg= "red")
data13.config(font=('Jersey M54 Font',100)) 
data13.pack()

data2.config(font=('Jersey M54 Font',200))
data2.pack()

get_text(root, eins, "spaces.txt")
get_text2(root, eins2, "spaces.txt")
root.mainloop()