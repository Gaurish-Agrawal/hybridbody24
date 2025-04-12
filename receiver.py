from flask import Flask, request
import threading
import pygame
import math

app = Flask(__name__)

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
sensor_targets = {d: 10000 for d in DIRECTIONS}
sensor_states = {d: {"length": 0, "color": (0, 255, 0)} for d in DIRECTIONS}

def lerp(a, b, t):
    return a + (b - a) * t

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
    if distance > 10000:
        return 10
    return int(80 - (min(distance, 10000) / 150))

@app.route("/update", methods=["POST"])
def update_data():
    data = request.get_json()
    print("Received:", data)
    direction = data.get("dir")
    dist = data.get("dist")
    if direction in sensor_targets and dist is not None and dist >= 0:
        sensor_targets[direction] = dist
    return {"status": "received"}

def run_flask():
    app.run(host='0.0.0.0', port=5000)

threading.Thread(target=run_flask, daemon=True).start()

pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("8-Directional Conical Pulse Visualizer")
clock = pygame.time.Clock()
cx, cy = width // 2, height // 2

pulse_timers = {d: 0 for d in DIRECTIONS}
bg_angle = 0

angle_lookup = {
    "N": -90,
    "NE": -45,
    "E": 0,
    "SE": 45,
    "S": 90,
    "SW": 135,
    "W": 180,
    "NW": -135
}

font = pygame.font.SysFont("Arial", 18, bold=False)

def get_offset(angle_deg, dist):
    angle_rad = math.radians(angle_deg)
    x = int(cx + dist * math.cos(angle_rad))
    y = int(cy + dist * math.sin(angle_rad))
    return x, y

def draw_background():
    global bg_angle
    grid_color = (50, 50, 50)
    for r in range(50, width // 2, 50):
        pygame.draw.circle(screen, grid_color, (cx, cy), r, 1)
    for i in range(0, 360, 30):
        angle = i + bg_angle
        end_pos = get_offset(angle, width // 2)
        pygame.draw.line(screen, grid_color, (cx, cy), end_pos, 1)
    bg_angle = (bg_angle + 0.1) % 360

def draw_text_with_outline(text, pos, font, text_color=(255,255,255), outline_color=(0,0,0), outline_width=2):
    base = font.render(text, True, text_color)
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx or dy:
                off = font.render(text, True, outline_color)
                off_rect = off.get_rect(center=(pos[0] + dx, pos[1] + dy))
                screen.blit(off, off_rect)
    base_rect = base.get_rect(center=pos)
    screen.blit(base, base_rect)

running = True
while running:
    dt = clock.tick(60) / 1000.0
    screen.fill((20, 20, 20))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_background()

    for d in DIRECTIONS:
        target_distance = sensor_targets[d]
        state = sensor_states[d]
        target_length = distance_to_length(target_distance)
        state["length"] = lerp(state["length"], target_length, 0.2)
        target_color = distance_to_color(target_distance)
        r = lerp(state["color"][0], target_color[0], 0.2)
        g = lerp(state["color"][1], target_color[1], 0.2)
        b = lerp(state["color"][2], target_color[2], 0.2)
        state["color"] = (int(r), int(g), int(b))

        pulse_timers[d] += dt * 2
        pulse_scale = 1 + 0.2 * math.sin(pulse_timers[d] * math.pi)
        final_length = int(state["length"] * pulse_scale)

        inner_offset = 50
        outer_offset = inner_offset + final_length
        half_angle = 20

        inner_left  = get_offset(angle_lookup[d] - half_angle, inner_offset)
        inner_right = get_offset(angle_lookup[d] + half_angle, inner_offset)
        outer_left  = get_offset(angle_lookup[d] - half_angle, outer_offset)
        outer_right = get_offset(angle_lookup[d] + half_angle, outer_offset)
        vertices = [inner_left, outer_left, outer_right, inner_right]

        #glows little bit
        for scale, alpha in [(1.15, 40), (1.25, 30), (1.35, 20)]:
            glow_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            glow_color = state["color"] + (alpha,)
            glow_vertices = []
            for vx, vy in vertices:
                glow_vx = int(cx + (vx - cx) * scale)
                glow_vy = int(cy + (vy - cy) * scale)
                glow_vertices.append((glow_vx, glow_vy))
            pygame.draw.polygon(glow_surface, glow_color, glow_vertices)
            screen.blit(glow_surface, (0, 0))
        
        pygame.draw.polygon(screen, state["color"], vertices)
        
        #distance and direction labels (w \n)
        text_color = "LightGray"
        label_offset = outer_offset + 50
        base_label_pos = get_offset(angle_lookup[d], label_offset)
        line_spacing = 25
        direction_pos = (base_label_pos[0], base_label_pos[1] - line_spacing // 2)
        distance_pos = (base_label_pos[0], base_label_pos[1] + line_spacing // 2)
        draw_text_with_outline(d, direction_pos, font, text_color)
        draw_text_with_outline(str(target_distance), distance_pos, font, text_color)
    
    pygame.display.flip()

pygame.quit()
