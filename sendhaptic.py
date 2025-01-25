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

root.mainloop()
