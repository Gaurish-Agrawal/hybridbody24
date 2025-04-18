from flask import Flask, request
import threading
import pygame
import math

app = Flask(__name__)

# Define 8 directions: cardinal and intercardinal
DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Initialize sensor target distances (large values mean "no obstacle")
sensor_targets = {dir: 10000 for dir in DIRECTIONS}
# Each sensor state holds the pulse "length" and its "color"
sensor_states = {dir: {"length": 0, "color": (0, 255, 0)} for dir in DIRECTIONS}

def lerp(a, b, t):
    """Linear interpolation between a and b by factor t."""
    return a + (b - a) * t

def distance_to_color(distance):
    """Smoothly convert a sensor distance value to a color (red for close, green for far)."""
    distance = max(20, distance)
    if distance < 30:
        return (255, 0, 0)  # Very close: Red
    elif distance < 50:
        ratio = (distance - 30) / 20
        return (255, int(ratio * 165), 0)  # Transition from Red to Orange
    elif distance < 100:
        ratio = (distance - 50) / 50
        r = 255 - int(ratio * 55)  # 255 to 200
        g = 165 + int(ratio * 90)  # 165 to 255
        return (r, g, 0)           # Orange to Yellow-Green
    elif distance < 1000:
        ratio = (distance - 100) / 900
        r = 200 - int(ratio * 200)  # 200 to 0
        return (r, 255, 0)          # Yellow-Green to Green
    else:
        return (0, 255, 0)          # Far away: Green

def distance_to_length(distance):
    """
    Convert sensor distance to a pulse length.
    Closer distances yield longer (more prominent) pulses.
    """
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

# Start Flask server in a separate thread
threading.Thread(target=run_flask, daemon=True).start()

# ------------------- Pygame Initialization -------------------
pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Conical Pulse Visualizer")
clock = pygame.time.Clock()

# Center coordinates
cx, cy = width // 2, height // 2

# Each sensor has its own pulse timer for a continuous sine-wave animation.
pulse_timers = {dir: 0 for dir in DIRECTIONS}

# Additional timer for the background rotation
bg_angle = 0

# Map each direction to its angle (in degrees)
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

# Initialize a font for labels
font = pygame.font.SysFont("Arial", 18)

def get_offset(angle_deg, dist):
    """
    Calculate the (x, y) coordinate offset from the center given an angle (in degrees)
    and a distance.
    """
    angle_rad = math.radians(angle_deg)
    x = int(cx + dist * math.cos(angle_rad))
    y = int(cy + dist * math.sin(angle_rad))
    return x, y

def draw_background():
    """Draw a rotating background grid (concentric circles and radial lines)."""
    global bg_angle
    # Set a light gray color for grid lines
    grid_color = (200, 200, 200)
    # Draw concentric circles
    for r in range(50, width//2, 50):
        pygame.draw.circle(screen, grid_color, (cx, cy), r, 1)
    # Draw radial lines; use bg_angle to rotate them slowly.
    for i in range(0, 360, 30):
        angle = i + bg_angle
        x, y = get_offset(angle, width//2)
        pygame.draw.line(screen, grid_color, (cx, cy), (x, y), 1)
    # Update the background rotation angle
    bg_angle = (bg_angle + 0.1) % 360

def draw_center():
    """Draw a pulsating central marker."""
    # Create a gentle pulsation effect using a sine wave
    pulse = 5 * math.sin(pygame.time.get_ticks() * 0.005) + 15
    pygame.draw.circle(screen, (0, 0, 0), (cx, cy), int(pulse))

running = True
while running:
    dt = clock.tick(60) / 1000.0  # Delta time in seconds
    screen.fill((30, 30, 30))  # Dark background for a modern look

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw the rotating background grid and the pulsating center
    draw_background()
    draw_center()

    # Process each of the 8 directions
    for direction in DIRECTIONS:
        target_distance = sensor_targets[direction]
        state = sensor_states[direction]

        # Update state: pulse length and color, smoothed via interpolation.
        target_length = distance_to_length(target_distance)
        state["length"] = lerp(state["length"], target_length, 0.2)

        target_color = distance_to_color(target_distance)
        r = lerp(state["color"][0], target_color[0], 0.2)
        g = lerp(state["color"][1], target_color[1], 0.2)
        b = lerp(state["color"][2], target_color[2], 0.2)
        state["color"] = (int(r), int(g), int(b))

        pulse_timers[direction] += dt * 2
        pulse_scale = 1 + 0.2 * math.sin(pulse_timers[direction] * math.pi)
        final_length = int(state["length"] * pulse_scale)

        # Parameters for the conical pulse shape.
        inner_offset = 50    # Offset to clear the center.
        outer_offset = inner_offset + final_length
        half_angle = 20      # Half of the cone's angular width.

        # Compute vertices for the pulse polygon.
        inner_left  = get_offset(angle_lookup[direction] - half_angle, inner_offset)
        inner_right = get_offset(angle_lookup[direction] + half_angle, inner_offset)
        outer_left  = get_offset(angle_lookup[direction] - half_angle, outer_offset)
        outer_right = get_offset(angle_lookup[direction] + half_angle, outer_offset)
        vertices = [inner_left, outer_left, outer_right, inner_right]

        # Draw a "glow" effect by drawing a larger, semi-transparent polygon behind the main one.
        glow_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        glow_color = state["color"] + (50,)  # Add alpha for transparency.
        # Increase size of glow by 10%
        glow_vertices = []
        for vx, vy in vertices:
            glow_vx = int(cx + (vx - cx) * 1.1)
            glow_vy = int(cy + (vy - cy) * 1.1)
            glow_vertices.append((glow_vx, glow_vy))
        pygame.draw.polygon(glow_surface, glow_color, glow_vertices)
        screen.blit(glow_surface, (0, 0))
        
        # Draw the main sensor pulse.
        pygame.draw.polygon(screen, state["color"], vertices)
        
        # Draw labels near the tip of each pulse, showing direction and the sensor value.
        label_text = f"{direction}: {target_distance}"
        label = font.render(label_text, True, (255,255,255))
        label_pos = get_offset(angle_lookup[direction], outer_offset + 20)
        screen.blit(label, label.get_rect(center=label_pos))
    
    pygame.display.flip()

pygame.quit()
