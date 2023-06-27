import tkinter as tk
import sqlite3
from PIL import ImageTk, Image
import datetime

# create the GUI window
root = tk.Tk()
root.title("Parking Spaces")

# connect to the database
conn = sqlite3.connect('kiosk/parking.db')

# define the function to update the values in the database
def update_values(delta, field):
    # retrieve the current capacity and occupancy values from the database
    c = conn.cursor()
    c.execute("SELECT capacity, occupancy FROM parking_spaces ORDER BY id DESC LIMIT 1")
    result = c.fetchone()
    capacity, occupancy = result

    # calculate the new values
    if field == 'capacity':
        new_capacity = capacity + delta
        new_occupancy = occupancy
    elif field == 'occupancy':
        new_capacity = capacity
        new_occupancy = occupancy + delta

    # update the values in the database
    c.execute("UPDATE parking_spaces SET capacity=?, occupancy=? WHERE id=1", (new_capacity, new_occupancy))
    conn.commit()

# define the function to update the display labels with the latest values from the database
def update_display():
    # retrieve the latest capacity and occupancy values from the database
    c = conn.cursor()
    c.execute("SELECT capacity, occupancy FROM parking_spaces ORDER BY id DESC LIMIT 1")
    result = c.fetchone()
    capacity, occupancy = result

    # update the display labels with the latest values
    capacity_label.config(text=f"Kapacita: {capacity}")
    occupancy_label.config(text=f"Obsadenosť: {occupancy}")

    # check if it's midnight and reset occupancy to 0 if the toggle button is activated
    now = datetime.datetime.now()
    if now.hour == 0 and midnight_reset.get() == False and toggle_var.get() == 1:
        midnight_reset.set(True)
        update_values(-occupancy, 'occupancy')
    elif now.hour != 0:
        midnight_reset.set(False)

    # schedule the next update in 1 second
    root.after(1000, update_display)

# load the image
image = Image.open("parking.jpg")
image = image.resize((100, 100))
photo = ImageTk.PhotoImage(image)

# create the label to display the image
image_label = tk.Label(root, image=photo)
image_label.pack()

# create the frame to hold the capacity widgets
capacity_frame = tk.Frame(root)
capacity_frame.pack()

# create the label to display the current capacity value
capacity_label = tk.Label(capacity_frame, text="")
capacity_label.pack(side=tk.LEFT)

# initialize the display label with the current capacity value from the database
c = conn.cursor()
c.execute("SELECT capacity FROM parking_spaces ORDER BY id DESC LIMIT 1")
result = c.fetchone()
capacity = result[0]
capacity_label.config(text=f"Capacity: {capacity}")

# create the buttons to increment or decrement the capacity value
increment_capacity_button = tk.Button(capacity_frame, text="+", command=lambda: update_values(1, 'capacity'))
increment_capacity_button.pack(side=tk.LEFT)

decrement_capacity_button = tk.Button(capacity_frame, text="-", command=lambda: update_values(-1, 'capacity'))
decrement_capacity_button.pack(side=tk.LEFT)

# create the frame to hold the occupancy widgets
occupancy_frame = tk.Frame(root)
occupancy_frame.pack()

# create the label to display the current occupancy value
occupancy_label = tk.Label(occupancy_frame, text="")
occupancy_label.pack(side=tk.LEFT)

# initialize the display label with the current occupancy value from the database
c = conn.cursor()
c.execute("SELECT occupancy FROM parking_spaces ORDER BY id DESC LIMIT 1")
result = c.fetchone()
occupancy = result[0]
occupancy_label.config(text=f"Occupancy: {occupancy}")

# create the buttons to increment or decrement the occupancy value
increment_occupancy_button = tk.Button(occupancy_frame, text="+", command=lambda: update_values(1, 'occupancy'))
increment_occupancy_button.pack(side=tk.RIGHT)

decrement_occupancy_button = tk.Button(occupancy_frame, text="-", command=lambda: update_values(-1, 'occupancy'))
decrement_occupancy_button.pack(side=tk.RIGHT)

# create the toggle button to reset occupancy to 0 at midnight
toggle_var = tk.IntVar()
toggle_button = tk.Checkbutton(root, text="Reset Occupancy at Midnight", variable=toggle_var)
toggle_button.pack()


# create a boolean variable to track if the midnight reset has occurred
midnight_reset = tk.BooleanVar()
midnight_reset.set(False)

# schedule the initial update of the display labels
update_display()

# start the GUI loop
root.mainloop()

# close the database connection when the GUI loop is finished
conn.close()