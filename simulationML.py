from flask import Flask, request
import threading
import pygame
import math
import joblib

app = Flask(__name__)

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
sensor_targets = {dir: 10000 for dir in DIRECTIONS}
sensor_states = {dir: {"length": 0, "color": (0, 255, 0)} for dir in DIRECTIONS}

try:
    model = joblib.load('xgb_model.joblib')
except Exception as e:
    print("WARNING: Failed to load model. Using dummy predictions.")
    model = None

def lerp(a, b, t): return a + (b - a) * t

def distance_to_color(distance):
    distance = max(20, distance)
    if distance < 30:
        return (255, 0, 0)
    elif distance < 50:
        ratio = (distance - 30) / 20
        return (255, int(ratio * 165), 0)
    elif distance < 100:
        ratio = (distance - 50) / 50
        r = 255 - int(ratio * 55)
        g = 165 + int(ratio * 90)
        return (r, g, 0)
    elif distance < 1000:
        ratio = (distance - 100) / 900
        r = 200 - int(ratio * 200)
        return (r, 255, 0)
    else:
        return (0, 255, 0)

def distance_to_length(distance):
    distance = max(20, distance)
    return int(80 - (min(distance, 10000) / 150))

@app.route("/sensorUpdate", methods=["GET"])
def update_data():
    print("Received request")
    print("Remote address:", request.remote_addr)
    print("Request args:", request.args)

    direction = request.args.get("channel")
    dist = request.args.get("distance")

    if direction not in sensor_targets:
        print(f"Invalid direction: {direction}")
        return {"error": "Invalid direction"}, 403

    try:
        dist = float(dist)
        if dist < 0:
            return {"error": "Negative distance"}, 403
    except Exception as e:
        return {"error": "Bad distance"}, 403

    print(f"Updated {direction} to distance {dist}")
    sensor_targets[direction] = dist
    return {"status": "received"}, 200

def run_flask():
    app.run(host='0.0.0.0', port=5000)

threading.Thread(target=run_flask, daemon=True).start()

# ------------------- Pygame -------------------
pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Conical Pulse Visualizer")
clock = pygame.time.Clock()
cx, cy = width // 2, height // 2
pulse_timers = {dir: 0 for dir in DIRECTIONS}
bg_angle = 0
angle_lookup = {
    "N": -90, "NE": -45, "E": 0, "SE": 45,
    "S": 90, "SW": 135, "W": 180, "NW": -135
}
font = pygame.font.SysFont("Arial", 18)

def get_offset(angle_deg, dist):
    angle_rad = math.radians(angle_deg)
    return int(cx + dist * math.cos(angle_rad)), int(cy + dist * math.sin(angle_rad))

def draw_background():
    global bg_angle
    for r in range(50, width//2, 50):
        pygame.draw.circle(screen, (200, 200, 200), (cx, cy), r, 1)
    for i in range(0, 360, 30):
        angle = i + bg_angle
        x, y = get_offset(angle, width//2)
        pygame.draw.line(screen, (200, 200, 200), (cx, cy), (x, y), 1)
    bg_angle = (bg_angle + 0.1) % 360

def draw_center():
    pulse = 5 * math.sin(pygame.time.get_ticks() * 0.005) + 15
    pygame.draw.circle(screen, (0, 0, 0), (cx, cy), int(pulse))

running = True
while running:
    dt = clock.tick(60) / 1000.0
    screen.fill((30, 30, 30))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    features = [sensor_targets["NE"], sensor_targets["E"], sensor_targets["SE"]]
    shape_pred = model.predict([features])[0] if model else 0
    label = f"Shape: {'Circular' if shape_pred == 0 else 'Rectangular' if shape_pred == 1 else 'N/A'}"
    screen.blit(font.render(label, True, (255, 255, 255)), (10, 10))

    draw_background()
    draw_center()

    for direction in DIRECTIONS:
        dist = sensor_targets[direction]
        state = sensor_states[direction]
        state["length"] = lerp(state["length"], distance_to_length(dist), 0.2)
        target_color = distance_to_color(dist)
        state["color"] = tuple(int(lerp(c1, c2, 0.2)) for c1, c2 in zip(state["color"], target_color))

        pulse_timers[direction] += dt * 2
        final_length = int(state["length"] * (1 + 0.2 * math.sin(pulse_timers[direction] * math.pi)))

        inner_offset = 50
        outer_offset = inner_offset + final_length
        half_angle = 20
        inner_left  = get_offset(angle_lookup[direction] - half_angle, inner_offset)
        inner_right = get_offset(angle_lookup[direction] + half_angle, inner_offset)
        outer_left  = get_offset(angle_lookup[direction] - half_angle, outer_offset)
        outer_right = get_offset(angle_lookup[direction] + half_angle, outer_offset)
        vertices = [inner_left, outer_left, outer_right, inner_right]

        glow_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        glow_color = state["color"] + (50,)
        glow_vertices = [(int(cx + (vx - cx) * 1.1), int(cy + (vy - cy) * 1.1)) for vx, vy in vertices]
        pygame.draw.polygon(glow_surface, glow_color, glow_vertices)
        screen.blit(glow_surface, (0, 0))
        pygame.draw.polygon(screen, state["color"], vertices)

        label_text = f"{direction}: {int(dist)}"
        label = font.render(label_text, True, (255,255,255))
        screen.blit(label, label.get_rect(center=get_offset(angle_lookup[direction], outer_offset + 20)))

    pygame.display.flip()

pygame.quit()
