from flask import Flask, request
import threading
import pygame
import time
import math

app = Flask(__name__)

DIRECTIONS = ["N", "E", "S", "W"]

sensor_targets = {dir: 10000 for dir in DIRECTIONS}
sensor_states = {dir: {"radius": 0, "color": (0, 255, 0)} for dir in DIRECTIONS}

def lerp(a, b, t):
    return a + (b - a) * t

def distance_to_color(distance):
    distance = max(20, distance)
    
    if distance < 30:
        return (255, 0, 0)  # Very close - Red
    elif distance < 50:
        # Transition from Red to Orange
        ratio = (distance - 30) / 20  # from 0 to 1
        return (255, int(ratio * 165), 0)
    elif distance < 100:
        # Transition from Orange to Yellow-Green
        ratio = (distance - 50) / 50  # from 0 to 1
        r = 255 - int(ratio * 55)     # 255 to 200
        g = 165 + int(ratio * 90)     # 165 to 255
        return (r, g, 0)
    elif distance < 1000:
        # Smooth transition from Yellow-Green to Green
        ratio = (distance - 100) / 900  # from 0 to 1
        r = 200 - int(ratio * 200)      # 200 to 0
        return (r, 255, 0)
    else:
        return (0, 255, 0)  # Very far - Green
    
    

def distance_to_radius(distance):
    distance = max(20, distance)
    if distance > 10000:
        return 10
    return int(80 - (min(distance, 10000) / 150))

@app.route("/update", methods=["POST"])
def update_data():
    data = request.get_json()
    print("Received:", data)
    direction = data.get("dir")
    dist = data.get("dist")

    if direction in sensor_targets and dist is not None:
        if dist >= 0:
            sensor_targets[direction] = dist

    return {"status": "received"}

def run_flask():
    app.run(host='0.0.0.0', port=5000)



threading.Thread(target=run_flask, daemon=True).start()



pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("4-Direction Proximity Visualizer")
clock = pygame.time.Clock()

cx, cy = width // 2, height // 2
pulse_timers = {dir: 0 for dir in DIRECTIONS}

angle_lookup = {
    "N": -90, 
    "E": 0, 
    "S": 90, 
    "W": 180
    }

def get_offset(angle_deg, dist=100):
    angle_rad = math.radians(angle_deg)
    x = int(cx + dist * math.cos(angle_rad))
    y = int(cy + dist * math.sin(angle_rad))
    return x, y

running = True
while running:
    dt = clock.tick(60) / 1000
    screen.fill((255, 255, 255))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.draw.circle(screen, (0, 0, 0), (cx, cy), 15)

    for direction in DIRECTIONS:
        target_distance = sensor_targets[direction]
        state = sensor_states[direction]

        target_radius = distance_to_radius(target_distance)
        state["radius"] = lerp(state["radius"], target_radius, 0.2)

        target_color = distance_to_color(target_distance)
        r = lerp(state["color"][0], target_color[0], 0.2)
        g = lerp(state["color"][1], target_color[1], 0.2)
        b = lerp(state["color"][2], target_color[2], 0.2)
        state["color"] = (int(r), int(g), int(b))

        #circles
        pulse_timers[direction] += dt * 2
        pulse_scale = 1 + 0.2 * math.sin(pulse_timers[direction] * math.pi)
        final_radius = int(state["radius"] * pulse_scale)

        x, y = get_offset(angle_lookup[direction], dist=100)
        pygame.draw.circle(screen, state["color"], (x, y), final_radius, 3)

    pygame.display.flip()

pygame.quit()
