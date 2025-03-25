import tkinter as tk
import requests
import threading
import time
import math

#adding pulse rings (circles)

CANVAS_WIDTH = 600
CANVAS_HEIGHT = 400

simulation_mode = False

person_x = CANVAS_WIDTH / 2
person_y = CANVAS_HEIGHT / 2
move_speed = 3

send_interval = 0.5  # delay (s)
last_send_time = 0

keys_pressed = set()

obstacles = []
drawing_obstacle = False
obstacle_start_x = None
obstacle_start_y = None
current_obstacle_id = None

candidate_directions = {
    "N":  (0, -1),
    "NE": (1, -1),
    "E":  (1, 0),
    "SE": (1, 1),
    "S":  (0, 1),
    "SW": (-1, 1),
    "W":  (-1, 0),
    "NW": (-1, -1)
}
for key, (dx, dy) in candidate_directions.items():
    norm = math.sqrt(dx**2 + dy**2)
    candidate_directions[key] = (dx / norm, dy / norm)

DETECTION_RADII = {
    0.25: 80,  # outermost ring
    0.5: 60,
    0.75: 40,
    1.0: 20    # innermost ring
}

#replace prev
DETECTION_RADII = {
    0.25: 40,  # outermost ring (was 80)
    0.5: 30,   # (was 60)
    0.75: 20,  # (was 40)
    1.0: 10    # innermost ring (was 20)
}

# Animation settings
PULSE_SPEED = 2  # pixels per frame
PULSE_INTERVAL = 33  # ms (approx 30 fps)
pulse_rings = {}  # Store active pulse animations

global display_pulse
display_pulse = False

def send_signal(command):
    arduino_url = "http://192.168.1.44/"
    try:
        full_url = arduino_url + command
        print(f"Sending: {command} --> {full_url}")
        response = requests.get(full_url)
        result_label.config(text=f"Signal '{command}' sent.")
    except Exception as e:
        result_label.config(text=f"Error: {e}")

def is_inside_obstacle(x, y):
    for (x1, y1, x2, y2, _) in obstacles:
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        if left <= x <= right and top <= y <= bottom:
            return True
    return False

def distance_to_nearest_obstacle(x, y):
    min_dist = float('inf')
    for (x1, y1, x2, y2, _) in obstacles:
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        if left <= x <= right and top <= y <= bottom:
            return 0
        dx = max(left - x, 0, x - right)
        dy = max(top - y, 0, y - bottom)
        dist = math.sqrt(dx**2 + dy**2)
        min_dist = min(min_dist, dist)
    return min_dist if min_dist != float('inf') else 1e6

def get_sensor_intensity(x, y, direction):
    dx, dy = candidate_directions[direction]
    max_range = DETECTION_RADII[0.25]
    check_dist = distance_to_nearest_obstacle(x + dx * max_range, y + dy * max_range)
    
    if check_dist > max_range:
        return 0.0
    
    for intensity, radius in sorted(DETECTION_RADII.items(), reverse=True):
        if check_dist <= radius:
            return intensity
    return 0.0

def create_pulse_ring(direction, intensity):
    if intensity == 0 or display_pulse==False: #not in danger or dont want
        return
    dx, dy = candidate_directions[direction]
    #range of intensities
    color_map = {1.0: "red", 0.75: "orange", 0.5: "yellow", 0.25: "green"}
    color = color_map[intensity]
    max_radius = DETECTION_RADII[intensity]
    ring_id = canvas.create_oval(
        person_x - 5, person_y - 5, person_x + 5, person_y + 5,
        outline=color, width=2, dash=(4, 4)
    )
    pulse_rings[(direction, intensity)] = {"id": ring_id, "radius": 0, "max_radius": max_radius, "dx": dx, "dy": dy}

def update_pulse_rings():
    to_remove = []
    for (direction, intensity), ring in list(pulse_rings.items()):
        ring["radius"] += PULSE_SPEED
        radius = ring["radius"]
        max_radius = ring["max_radius"]
        dx, dy = ring["dx"], ring["dy"]
        center_x = person_x + dx * radius
        center_y = person_y + dy * radius
        
        # Update ring position
        canvas.coords(ring["id"],
                      center_x - radius, center_y - radius,
                      center_x + radius, center_y + radius)
        
        # Check if pulse hits an obstacle or reaches max radius
        if radius >= max_radius or distance_to_nearest_obstacle(center_x, center_y) < radius:
            canvas.itemconfig(ring["id"], outline="grey")  # Dim when hitting obstacle or max
            to_remove.append((direction, intensity))
        
    for key in to_remove:
        canvas.delete(pulse_rings[key]["id"])
        del pulse_rings[key]

def toggle_mode():
    global simulation_mode, person_x, person_y, keys_pressed, pulse_rings
    simulation_mode = not simulation_mode
    if simulation_mode:
        mode_button.config(text="Switch to Edit Mode")
        info_label.config(text="Simulation Mode: Use arrow keys for movement")
        person_x = CANVAS_WIDTH / 2
        person_y = CANVAS_HEIGHT / 2
        canvas.coords(person_dot, person_x-5, person_y-5, person_x+5, person_y+5)
        canvas.unbind("<ButtonPress-1>")
        canvas.unbind("<B1-Motion>")
        canvas.unbind("<ButtonRelease-1>")
        root.bind("<KeyPress>", on_key_press)
        root.bind("<KeyRelease>", on_key_release)
        root.focus_set()
    else:
        mode_button.config(text="Switch to Simulation Mode")
        info_label.config(text="Edit Mode: Click and drag to add obstacles")
        root.unbind("<KeyPress>")
        root.unbind("<KeyRelease>")
        keys_pressed.clear()
        canvas.coords(person_dot, -10, -10, -10, -10)
        canvas.bind("<ButtonPress-1>", start_obstacle)
        canvas.bind("<B1-Motion>", update_obstacle)
        canvas.bind("<ButtonRelease-1>", end_obstacle)
        update_belt_display({dir: 0.0 for dir in candidate_directions})
        for ring in pulse_rings.values():
            canvas.delete(ring["id"])
        pulse_rings.clear()

def start_obstacle(event):
    global drawing_obstacle, obstacle_start_x, obstacle_start_y, current_obstacle_id
    if simulation_mode:
        return
    drawing_obstacle = True
    obstacle_start_x, obstacle_start_y = event.x, event.y
    current_obstacle_id = canvas.create_rectangle(obstacle_start_x, obstacle_start_y, event.x, event.y,
                                                  fill="grey", stipple="gray25", outline="black")

def update_obstacle(event):
    if simulation_mode or not drawing_obstacle or current_obstacle_id is None:
        return
    canvas.coords(current_obstacle_id, obstacle_start_x, obstacle_start_y, event.x, event.y)

def end_obstacle(event):
    global drawing_obstacle, obstacles, current_obstacle_id
    if simulation_mode or not drawing_obstacle or current_obstacle_id is None:
        return
    x1, y1, x2, y2 = canvas.coords(current_obstacle_id)
    obstacles.append((x1, y1, x2, y2, current_obstacle_id))
    drawing_obstacle = False
    current_obstacle_id = None

def clear_obstacles():
    global obstacles
    for (_, _, _, _, rect_id) in obstacles:
        canvas.delete(rect_id)
    obstacles = []

def on_key_press(event):
    if event.keysym in ("Up", "Down", "Left", "Right"):
        keys_pressed.add(event.keysym)

def on_key_release(event):
    if event.keysym in keys_pressed:
        keys_pressed.remove(event.keysym)

def update_person_position():
    global person_x, person_y
    dx, dy = 0, 0
    if "Up" in keys_pressed:
        dy -= move_speed
    if "Down" in keys_pressed:
        dy += move_speed
    if "Left" in keys_pressed:
        dx -= move_speed
    if "Right" in keys_pressed:
        dx += move_speed
    new_x = person_x + dx
    new_y = person_y + dy
    new_x = max(0, min(CANVAS_WIDTH, new_x))
    new_y = max(0, min(CANVAS_HEIGHT, new_y))
    if not is_inside_obstacle(new_x, new_y):
        person_x, person_y = new_x, new_y
        canvas.coords(person_dot, person_x-5, person_y-5, person_x+5, person_y+5)

belt_motors = {}

def create_belt_display():
    center_x = BELT_CANVAS_SIZE / 2
    center_y = BELT_CANVAS_SIZE / 2
    belt_radius = 60
    motor_radius = 15
    angle_map = {
        "N": -90,
        "NE": -45,
        "E": 0,
        "SE": 45,
        "S": 90,
        "SW": 135,
        "W": 180,
        "NW": -135
    }
    for direction, angle_deg in angle_map.items():
        angle_rad = math.radians(angle_deg)
        x = center_x + belt_radius * math.cos(angle_rad)
        y = center_y + belt_radius * math.sin(angle_rad)
        item = belt_canvas.create_oval(x - motor_radius, y - motor_radius,
                                      x + motor_radius, y + motor_radius,
                                      fill="#555555", outline="black", width=2)
        belt_motors[direction] = item

def update_belt_display(intensity_map):
    off_color = "#555555"
    for direction, item in belt_motors.items():
        intensity = intensity_map[direction]
        if intensity > 0:
            red_val = int(85 + intensity * (255 - 85))
            active_color = f"#{red_val:02x}0000"
            belt_canvas.itemconfig(item, fill=active_color)
        else:
            belt_canvas.itemconfig(item, fill=off_color)

def intensity_to_id(i):
    if i < 0 or i > 1:
        raise ValueError("Intensity must be between 0 and 1.")
    if 0 <= i <= 0.25:
        return 50
    elif 0.25 < i <= 0.5:
        return 49
    elif 0.5 < i <= 0.75:
        return 48
    elif 0.75 < i <= 1.0:
        return 47

def simulation_loop():
    global last_send_time
    if simulation_mode:
        update_person_position()
        intensity_map = {}
        for direction in candidate_directions:
            intensity = get_sensor_intensity(person_x, person_y, direction)
            intensity_map[direction] = intensity
            # Create new pulse if not already pulsing for this intensity
            if intensity > 0 and (direction, intensity) not in pulse_rings:
                create_pulse_ring(direction, intensity)
        
        update_pulse_rings()
        info_text = f"Active Sensors: {sum(1 for i in intensity_map.values() if i > 0)}"
        info_label.config(text=info_text, font=("Helvetica", 28, "bold"))
        update_belt_display(intensity_map)
        
        now = time.time()
        if now - last_send_time > send_interval:
            for direction, intensity in intensity_map.items():
                if intensity > 0:
                    command_str = f"{direction}?value={intensity_to_id(intensity)}"
                    threading.Thread(target=send_signal, args=(command_str,), daemon=True).start()
            last_send_time = now
    root.after(PULSE_INTERVAL, simulation_loop)

root = tk.Tk()
root.title("Indoor Environment Haptic Simulation")
root.geometry("1000x600")
root.configure(bg="#F5F5F5")
root.resizable(False, False)

main_frame = tk.Frame(root, bg="#F5F5F5")
main_frame.pack(padx=30, pady=30)

canvas = tk.Canvas(main_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#FFFFFF", bd=0, highlightthickness=0)
canvas.grid(row=0, column=0, padx=(0,30), pady=10)

BELT_CANVAS_SIZE = 200
belt_canvas = tk.Canvas(main_frame, width=BELT_CANVAS_SIZE, height=BELT_CANVAS_SIZE, bg="#F5F5F5", bd=0, highlightthickness=0)
belt_canvas.grid(row=0, column=1, padx=(30,0), pady=10)

person_dot = canvas.create_oval(-10, -10, -10, -10, fill="#007ACC", outline="black", width=1.5)

canvas.bind("<ButtonPress-1>", start_obstacle)
canvas.bind("<B1-Motion>", update_obstacle)
canvas.bind("<ButtonRelease-1>", end_obstacle)

control_frame = tk.Frame(root, bg="#F5F5F5")
control_frame.pack(pady=20)

mode_button = tk.Button(control_frame, text="Switch to Simulation Mode", font=("Helvetica", 14), relief="flat", bg="#FFFFFF", bd=1, padx=15, pady=8, command=toggle_mode)
mode_button.grid(row=0, column=0, padx=20)

clear_button = tk.Button(control_frame, text="Clear Obstacles", font=("Helvetica", 14), relief="flat", bg="#FFFFFF", bd=1, padx=15, pady=8, command=clear_obstacles)
clear_button.grid(row=0, column=1, padx=20)

# pulse on/off
def toggle_mode():
    #print("here!")
    global display_pulse
    display_pulse = not display_pulse
    toggle_button.config(text="Pulse ON" if display_pulse else "Pulse OFF")

toggle_button = tk.Button(control_frame, text="OFF", font=("Helvetica", 14), bg="lightgray", padx=20, pady=10, command=toggle_mode)
toggle_button.grid(row=0, column=2, padx=20)



info_label = tk.Label(root, text="Edit Mode: Click and drag to add obstacles", font=("Helvetica", 28, "bold"), bg="black")
info_label.pack(pady=(10,5))

result_label = tk.Label(root, text="Command result here", font=("Helvetica", 14), bg="#F5F5F5")
result_label.pack(pady=(0,10))


create_belt_display()
simulation_loop()
root.mainloop()
