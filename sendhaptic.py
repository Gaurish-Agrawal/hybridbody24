#haptics with zoning gui for sumulation

import tkinter as tk
import requests
import threading
import time

arduino_url = "http://192.168.1.125/"
send_interval = 0.1 #delay
last_send_time = 0

def send_signal(command):
    try:
        full_url = arduino_url + command
        print(f"Attempting to send command: {command}")
        print(f"Full URL: {full_url}")
        response = requests.get(full_url)
        print(f"Sent command: {command}, Response: {response.text}")
        result_label.config(text=f"Signal '{command}' sent successfully")
    except Exception as e:
        result_label.config(text=f"Error sending command: {e}")
        print(f"Error sending command: {e}")

def get_zone(x, y):

    #green (inner) (safe)
    if 150 <= x <= 450 and 150 <= y <= 250:
        return 4
    
    elif 100 <= x <= 500 and 100 <= y <= 300:
        return 3
    
    elif 50 <= x <= 550 and 50 <= y <= 350:
        return 2

    #red (outermost)
    return 1

def zone_to_intensity(zone):

    if zone == 4:
        return 0.00 #green
    elif zone == 3:
        return 0.20 #yellow
    elif zone == 2:
        return 0.55 #orange
    else:            
        return 1.00 #red


def get_direction(x, y):

    dist_left   = x
    dist_top    = y
    dist_right  = CANVAS_WIDTH  - x
    dist_bottom = CANVAS_HEIGHT - y
    
    nearest_lt = min(dist_left, dist_top)
    nearest_rb = min(dist_right, dist_bottom)
    
    if nearest_lt < nearest_rb:
        return "RIGHT"
    else:
        return "LEFT"


def on_mouse_move(event):
    global last_send_time

    mouse_x = event.x
    mouse_y = event.y

    R = 5
    canvas.coords(mouse_dot, mouse_x - R, mouse_y - R, mouse_x + R, mouse_y + R)

    zone = get_zone(mouse_x, mouse_y)
    intensity = zone_to_intensity(zone)
    direction = get_direction(mouse_x, mouse_y)

    zone_name = {1:"RED", 2:"ORANGE", 3:"YELLOW", 4:"GREEN"}[zone]
    info_label.config(
        text=f"Zone: {zone_name} | Direction: {direction} | Intensity: {intensity:.2f}"
    )

    if zone == 4:
        return

    now = time.time()
    if now - last_send_time > send_interval:
        last_send_time = now
        command_str = f"{direction}?value={intensity:.2f}"
        threading.Thread(target=send_signal, args=(command_str,), daemon=True).start()


root = tk.Tk()
root.title("Four-Zone Haptic Demo (Fixed Intensities)")
root.geometry("600x500") #fixed
root.resizable(False, False)

CANVAS_WIDTH  = 600
CANVAS_HEIGHT = 400

canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
canvas.pack()

canvas.create_rectangle(
    0, 0, CANVAS_WIDTH, CANVAS_HEIGHT,
    fill="#F8D0D8", outline=""
)

canvas.create_rectangle(
    50, 50, 550, 350,
    fill="#FFE5CA", outline=""
)

canvas.create_rectangle(
    100, 100, 500, 300,
    fill="#FFFFCC", outline=""
)

# green zone (safe)
canvas.create_rectangle(
    150, 150, 450, 250,
    fill="#DAF7DC", outline=""
)

mouse_dot = canvas.create_oval(0, 0, 10, 10, fill="blue", outline="black")
info_label = tk.Label(root, text="Move mouse inside the canvas", font=("Arial", 12))
info_label.pack(pady=5)
result_label = tk.Label(root, text="Command result here", font=("Arial", 12))
result_label.pack(pady=5)

canvas.bind("<Motion>", on_mouse_move)

root.mainloop()

"""
import tkinter as tk
import requests
import cv2
import tkinter as tk
import requests
import threading

arduino_url = "http://192.168.1.125/"

def send_signal(command):
    try:
        full_url = arduino_url + command

        print(full_url)

        print(f"Attempting to send command: {command}")
        print(f"Full URL: {full_url}")

        response = requests.get(full_url)
        print(f"Sent command: {command}, Response: {response.text}")

        result_label.config(text=f"Signal '{command}' sent successfully")
        
    except Exception as e:
        result_label.config(text=f"Error sending command: {e}")
        print(f"Error sending command: {e}")



def handle_button_click(command):
    threading.Thread(target=send_signal, args=(command,), daemon=True).start()

root = tk.Tk()
root.title("Haptic Motor Control")

right_button = tk.Button(root, text="RIGHT", command=lambda: handle_button_click("RIGHT"), height=2, width=10)
right_button.grid(row=0, column=0, padx=20, pady=20)
left_button = tk.Button(root, text="LEFT", command=lambda: handle_button_click("LEFT"), height=2, width=10)
left_button.grid(row=0, column=1, padx=20, pady=20)
result_label = tk.Label(root, text="Click a button to send a signal", font=("Arial", 12))
result_label.grid(row=1, column=0, columnspan=2, pady=10)
"""
root.mainloop()
