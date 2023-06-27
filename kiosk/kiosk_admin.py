from tkinter import *

def get_text(root, val1, val2, name):
    try:
        with open(name, "r") as f:
            lines = f.readlines()
            nums = lines[1].split()
            val1.set(nums[1])
            val2.set(nums[2])
    except IOError as e:
        print(e)
    else:
        root.after(1000, lambda: get_text(root, val1, val2, name))

def increment_val():
    with open("kiosk/spaces.txt", "r") as f:
        lines = f.readlines()
        nums = lines[1].split()
    nums[1] = str(int(nums[1]) + 1)
    nums[2] = str(int(nums[2]) - 1)
    nums_str = "        ".join([str(n) for n in nums])
    lines[1] = nums_str + "\n"
    with open("kiosk/spaces.txt", "w") as f:
        f.writelines(lines)

def decrement_val():
    with open("kiosk/spaces.txt", "r") as f:
        lines = f.readlines()
        nums = lines[1].split()
    nums[1] = str(int(nums[1]) - 1)
    nums[2] = str(int(nums[2]) + 1)
    nums_str = "        ".join([str(n) for n in nums])
    lines[1] = nums_str + "\n"
    with open("kiosk/spaces.txt", "w") as f:
        f.writelines(lines)

def save_kapacita(new_val):
    with open("kiosk/spaces.txt", "r") as f:
        lines = f.readlines()
        nums = lines[1].split()
    nums[0] = str(int(new_val))
    nums_str = "        ".join([str(n) for n in nums])
    lines[1] = nums_str + "\n"
    with open("kiosk/spaces.txt", "w") as f:
        f.writelines(lines)

def quit_win():
    root.destroy()

#root = Tk()
#root.minsize(1080,1920)
#root.attributes('-fullscreen', True)
#root.configure(bg='white')

root = Tk()
root.geometry('700x350')
root.configure(bg='white')


Kapacita = StringVar()
Volne = StringVar()
Obsadene = StringVar()


kapacita_label = Label(root, text="Kapacita:", bg='white', fg="black", font=('Jersey M54 Font', 20))
kapacita_label.grid(row=2, column=0, padx=10, pady=10)

kapacita_entry = Entry(root, textvariable=Kapacita, bg='white', fg="black", font=('Jersey M54 Font', 20))
kapacita_entry.grid(row=2, column=1, padx=0, pady=10)

save_button = Button(root, text="Save", command=lambda: save_kapacita(Kapacita.get()), height=3, width=10)
save_button.grid(row=2, column=3, padx=10, pady=10)

decrement_button2 = Button(root, text='▼', command=lambda: decrement_val(), height=2, width=5)
decrement_button2.grid(row=0, column=3, padx=10, pady=10)

increment_button2 = Button(root, text='▲', command=lambda: increment_val(), height=2, width=5)
increment_button2.grid(row=0, column=2, padx=10, pady=10)

data1 = Label(root, text="Volne:", bg='white', fg="green", font=('Jersey M54 Font', 20))
data1.grid(row=0, column=0, padx=10, pady=10)

data2 = Label(root, text="Obsadene:", bg='white', fg="red", font=('Jersey M54 Font', 20))
data2.grid(row=1, column=0, padx=10, pady=10)

val_label = Label(root, textvariable=Volne, bg='white', fg="green", font=('Jersey M54 Font', 50))
val_label.grid(row=0, column=1, padx=10, pady=10)

val2_label = Label(root, textvariable=Obsadene, bg='white', fg="red", font=('Jersey M54 Font', 50))
val2_label.grid(row=1, column=1, padx=10, pady=10)



get_text(root, Volne, Obsadene, "kiosk/spaces.txt")

root.mainloop()