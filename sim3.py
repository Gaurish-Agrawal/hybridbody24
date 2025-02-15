import tkinter as tk
import requests
import threading
import time
import math

arduino_url = "http://192.168.1.125/"
send_interval = 0.1  # Minimum seconds between sending commands
last_send_time = 0

CANVAS_WIDTH = 600
CANVAS_HEIGHT = 400

simulation_mode = False

# Simulated person (blue dot)
person_x = CANVAS_WIDTH / 2
person_y = CANVAS_HEIGHT / 2
move_speed = 3  
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

safe_threshold = 50    # If closer than this to an obstacle/wall, intensity increases.
candidate_step = 20    # Pixels to test candidate moves


def send_signal(command):
    """Send an HTTP GET command (non-blocking) to the haptic device."""
    try:
        full_url = arduino_url + command
        #print(f"Sending: {command} -> {full_url}")
        response = requests.get(full_url)
        result_label.config(text=f"Signal '{command}' sent.")
    except Exception as e:
        result_label.config(text=f"Error: {e}")

def compute_intensity(distance):
    """Return haptic intensity [0,1] based on distance to obstacle/wall."""
    if distance >= safe_threshold:
        return 0.0
    else:
        return round((safe_threshold - distance) / safe_threshold, 2)

def is_inside_obstacle(x, y):
    """Return True if point (x, y) is inside any obstacle."""
    for (x1, y1, x2, y2, _) in obstacles:
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        if left <= x <= right and top <= y <= bottom:
            return True
    return False

def distance_to_obstacle(x, y):
    min_dist = float('inf')
    for (x1, y1, x2, y2, _) in obstacles:
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        if left <= x <= right and top <= y <= bottom:
            return 0
        dx = 0
        dy = 0
        if x < left:
            dx = left - x
        elif x > right:
            dx = x - right
        if y < top:
            dy = top - y
        elif y > bottom:
            dy = y - bottom
        dist = math.sqrt(dx**2 + dy**2)
        if dist < min_dist:
            min_dist = dist
    return min_dist if min_dist != float('inf') else 1e6

def get_min_distance(x, y):
    boundary_dist = min(x, CANVAS_WIDTH - x, y, CANVAS_HEIGHT - y)
    obs_dist = distance_to_obstacle(x, y)
    return min(boundary_dist, obs_dist)

def get_recommended_direction_simulation(x, y):
    """
    Returns the safe (recommended) direction: the candidate direction that leads to the highest clearance.
    """
    best_dir = None
    best_distance = -1
    for dir_name, (dx, dy) in candidate_directions.items():
        new_x = x + candidate_step * dx
        new_y = y + candidate_step * dy
        if not (0 <= new_x <= CANVAS_WIDTH and 0 <= new_y <= CANVAS_HEIGHT):
            continue
        if is_inside_obstacle(new_x, new_y):
            continue
        dist = get_min_distance(new_x, new_y)
        if dist > best_distance:
            best_distance = dist
            best_dir = dir_name
    return best_dir if best_dir is not None else "None"

def get_crash_direction_simulation(x, y):
    """
    Returns the crash direction: the candidate direction with the minimum clearance,
    i.e. the direction in which a collision is most imminent.
    """
    crash_dir = None
    min_distance = float('inf')
    for dir_name, (dx, dy) in candidate_directions.items():
        new_x = x + candidate_step * dx
        new_y = y + candidate_step * dy
        if not (0 <= new_x <= CANVAS_WIDTH and 0 <= new_y <= CANVAS_HEIGHT):
            distance = 0
        else:
            distance = get_min_distance(new_x, new_y)
        if distance < min_distance:
            min_distance = distance
            crash_dir = dir_name
    return crash_dir, min_distance

def toggle_mode():
    global simulation_mode, person_x, person_y, keys_pressed
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
        update_belt_display("None", 0)

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

#Arrow Keys
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

#Belt visual
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

def update_belt_display(active_direction, intensity):
    off_color = "#555555"
    for direction, item in belt_motors.items():
        if direction == active_direction and intensity > 0:
            red_val = int(85 + intensity * (255 - 85))
            active_color = f"#{red_val:02x}0000"
            belt_canvas.itemconfig(item, fill=active_color)
        else:
            belt_canvas.itemconfig(item, fill=off_color)

def simulation_loop():
    global last_send_time
    if simulation_mode:
        update_person_position()
        safe_dir = get_recommended_direction_simulation(person_x, person_y)
        crash_dir, crash_distance = get_crash_direction_simulation(person_x, person_y)
        current_min_dist = get_min_distance(person_x, person_y)
        intensity = compute_intensity(current_min_dist)
        # Display safe direction in larger, bold text.
        info_text = f"Safe Direction: {safe_dir}    Intensity: {intensity}"
        info_label.config(text=info_text, font=("Helvetica", 28, "bold"))
        update_belt_display(crash_dir, intensity)
        if intensity > 0 and crash_dir != "None":
            now = time.time()
            if now - last_send_time > send_interval:
                last_send_time = now
                command_str = f"{crash_dir}?value={intensity:.2f}"
                threading.Thread(target=send_signal, args=(command_str,), daemon=True).start()
    root.after(33, simulation_loop)

#ui
root = tk.Tk()
root.title("Indoor Environment Haptic Simulation")
root.geometry("1000x600")
root.configure(bg="#F5F5F5")
root.resizable(False, False)

# Main frame
main_frame = tk.Frame(root, bg="#F5F5F5")
main_frame.pack(padx=30, pady=30)
# Simulation Canvas (left)
canvas = tk.Canvas(main_frame, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="#FFFFFF", bd=0, highlightthickness=0)
canvas.grid(row=0, column=0, padx=(0,30), pady=10)
# Belt Display (right) – no extra background, just the circles
BELT_CANVAS_SIZE = 200
belt_canvas = tk.Canvas(main_frame, width=BELT_CANVAS_SIZE, height=BELT_CANVAS_SIZE, bg="#F5F5F5", bd=0, highlightthickness=0)
belt_canvas.grid(row=0, column=1, padx=(30,0), pady=10)

#person (init hide)
person_dot = canvas.create_oval(-10, -10, -10, -10, fill="#007ACC", outline="black", width=1.5)

canvas.bind("<ButtonPress-1>", start_obstacle)
canvas.bind("<B1-Motion>", update_obstacle)
canvas.bind("<ButtonRelease-1>", end_obstacle)

#grids/allignment
control_frame = tk.Frame(root, bg="#F5F5F5")
control_frame.pack(pady=20)

mode_button = tk.Button(control_frame, text="Switch to Simulation Mode", font=("Helvetica", 14), relief="flat", bg="#FFFFFF", bd=1, padx=15, pady=8, command=toggle_mode)
mode_button.grid(row=0, column=0, padx=20)

clear_button = tk.Button(control_frame, text="Clear Obstacles", font=("Helvetica", 14), relief="flat", bg="#FFFFFF", bd=1, padx=15, pady=8, command=clear_obstacles)
clear_button.grid(row=0, column=1, padx=20)

info_label = tk.Label(root, text="Edit Mode: Click and drag to add obstacles", font=("Helvetica", 28, "bold"), bg="black")
info_label.pack(pady=(10,5))

result_label = tk.Label(root, text="Command result here", font=("Helvetica", 14), bg="#F5F5F5")
result_label.pack(pady=(0,10))

create_belt_display()
simulation_loop()

root.mainloop()
