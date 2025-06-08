import pygame
import math
import numpy as np

def lerp(t, start, end):
    return (1 - t) * start + t * end

def draw_reverb_visualizer(surface, x, y, width, height, reverb_params):
    size, mix, tone, tail, delay = reverb_params
    
    # Draw background
    pygame.draw.rect(surface, (40, 0, 80), (x, y, width, height))
    
    # Calculate decay curve points
    points = []
    num_points = 100
    for i in range(num_points):
        t = i / (num_points - 1)
        # Exponential decay based on reverb parameters
        decay = math.exp(-t * (1 + size * 2))  # Size affects decay rate
        amplitude = mix * decay  # Mix affects overall amplitude
        x_pos = x + t * width
        y_pos = y + height/2 - (amplitude * height/2)
        points.append((x_pos, y_pos))
    
    # Draw decay curve
    if len(points) > 1:
        pygame.draw.lines(surface, (200, tone*255, 255), False, points, 2)
    
    # Draw delay line
    delay_x = x + (delay * width)
    pygame.draw.line(surface, (255, 255, 255), (delay_x, y), (delay_x, y + height), 1)
    
    # Draw tail line
    tail_x = x + (tail * width)
    pygame.draw.line(surface, (255, 100, 100), (tail_x, y), (tail_x, y + height), 1)
    
    # Draw labels
    font = pygame.font.SysFont(None, 20)
    delay_text = font.render(f"Delay", True, (255, 255, 255))
    tail_text = font.render(f"Tail", True, (255, 100, 100))
    surface.blit(delay_text, (delay_x - 20, y - 20))
    surface.blit(tail_text, (tail_x - 15, y - 20))



import pygame

def draw_eq_visualizer(surface, x, y, width, height, eq_params):
    """
    Draw a 7-band EQ visualizer on the given surface.
    
    Parameters:
    - surface: pygame.Surface to draw on.
    - x, y: Top-left corner coordinates of the visualizer.
    - width, height: Total width and height of the visualizer.
    - eq_params: List of 7 float values (0.0 to 1.0) representing EQ boost levels
                 for the bands: [100Hz, 200Hz, 400Hz, 800Hz, 1.6kHz, 3.2kHz, 6.4kHz].
    """
    assert len(eq_params) == 7, "eq_params must contain 7 values."
    
    # Frequency labels for each band
    frequencies = ['100Hz', '200Hz', '400Hz', '800Hz', '1.6kHz', '3.2kHz', '6.4kHz']
    
    bar_spacing = 10
    num_bands = len(eq_params)
    bar_width = (width - (num_bands - 1) * bar_spacing) / num_bands
    max_bar_height = height

    font = pygame.font.SysFont('Arial', 14)
    
    for i, value in enumerate(eq_params):
        # Clamp value just in case
        value = max(0.0, min(1.0, value))
        
        # Calculate bar height and position
        bar_height = value * max_bar_height
        bar_x = x + i * (bar_width + bar_spacing)
        bar_y = y + (max_bar_height - bar_height)

        # Color changes from red (low) to green (high)
        color = (
            int((1 - value) * 255),  # Red decreases
            int(value * 255),        # Green increases
            50                      # Constant blue for balance
        )

        # Draw bar
        pygame.draw.rect(surface, color, (bar_x, bar_y, bar_width, bar_height))

        # Draw frequency label centered under the bar
        label = font.render(frequencies[i], True, (255, 255, 255))
        label_rect = label.get_rect(center=(bar_x + bar_width / 2, y + height + 12))
        surface.blit(label, label_rect)

        # Optional: draw numeric gain value above the bar
        gain_label = font.render(f"{value:.2f}", True, (255,255,255))
        gain_rect = gain_label.get_rect(center=(bar_x + bar_width / 2, bar_y - 10))
        surface.blit(gain_label, gain_rect)
def draw_delay_visualizer(surface, x, y, width, height, delay_params):
    """
    Visualizes delay parameters using repeated echo-like circles and stereo spread.
    
    Parameters:
    - surface: pygame.Surface to draw on.
    - x, y: Top-left corner of the visualizer area.
    - width, height: Width and height of the visualizer box.
    - delay_params: [mix (0-1), delay_in_ms (0-2000), spread (0-1)]
    """
    mix, delay_ms, spread = delay_params
    delay_ms = round(delay_ms * 2000)
    # Clamp values
    mix = max(0.0, min(1.0, mix))
    delay_ms = max(0.0, min(2000.0, delay_ms))
    spread = max(0.0, min(1.0, spread)) 

    # Draw background box
    pygame.draw.rect(surface, (20, 20, 40), (x, y, width, height), border_radius=10)
    
    center_x = x + width // 2
    center_y = y + height // 2

    # Number of echoes based on delay time
    num_echoes = max(2, int(delay_ms / 200))
    
    # Base radius for visual echo
    base_radius = min(width, height) // 10

    for i in range(num_echoes):
        # Opacity and size scale with mix and echo index
        fade = (1 - (i / num_echoes)) * mix
        radius = base_radius + i * 5
        alpha = int(fade * 255)

        # Spread control: wider spread means wider L/R echo offsets
        spread_offset = int((i / num_echoes) * spread * (width // 3))

        # Create a transparent surface for alpha blending
        echo_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Echo colors fade and spread over time
        color = (100, 200, 255, alpha)

        # Draw left and right echoes
        pygame.draw.circle(echo_surface, color, (center_x - spread_offset - x, center_y - y), radius)
        pygame.draw.circle(echo_surface, color, (center_x + spread_offset - x, center_y - y), radius)

        surface.blit(echo_surface, (x, y))

    # Draw central "original" pulse
    pygame.draw.circle(surface, (255, 255, 255), (center_x, center_y), base_radius // 2)
    
    # Label text
    font = pygame.font.SysFont("Arial", 14)
    labels = [
        f"Mix: {mix:.2f}",
        f"Delay: {int(delay_ms)} ms",
        f"Spread: {spread:.2f}"
    ]
    for i, text in enumerate(labels):
        label_surface = font.render(text, True, (180, 180, 180))
        surface.blit(label_surface, (x + 10, y + 10 + i * 18))


def draw_compressor_visualizer(surface, x, y, width, height, params):
    """
    Draws a visual representation of a compressor's behavior.

    Parameters:
    - surface: pygame.Surface to draw on
    - x, y: Top-left coordinates of the visualizer
    - width, height: Width and height of the visualizer
    - params: [threshold (dB), ratio (>=1), attack_ms, release_ms]
    """
    threshold, ratio, attack_ms, release_ms = params

    # Clamp and prepare values
    threshold = lerp(threshold, -60, 0)
    ratio = lerp(ratio, 1, 20)
    attack_ms = lerp(attack_ms, 0.1, 200)
    release_ms = lerp(release_ms, 10, 2000)

    # Draw background box
    pygame.draw.rect(surface, (25, 25, 35), (x, y, width, height), border_radius=10)

    # Fonts
    font = pygame.font.SysFont("Arial", 14)
    label_color = (255,255,255)

    # === Draw compression curve ===
    curve_color = (255, 242, 0)
    line_thickness = 2

    curve_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    for i in range(width):
        # Simulate input level from -60 dB (left) to 0 dB (right)
        input_db = -60 + (i / width) * 60

        if input_db < threshold:
            output_db = input_db
        else:
            output_db = threshold + (input_db - threshold) / ratio

        # Map output dB to screen height
        input_y = height - ((input_db + 60) / 60) * height
        output_y = height - ((output_db + 60) / 60) * height

        pygame.draw.line(curve_surface, curve_color, (i, height), (i, output_y), line_thickness)

    surface.blit(curve_surface, (x, y))

    # === Threshold Line ===
    thresh_y = y + height - ((threshold + 60) / 60) * height
    pygame.draw.line(surface, (255, 100, 100), (x, thresh_y), (x + width, thresh_y), 2)

    # === Attack / Release Bars ===
    bar_width = 12
    attack_height = min(height, int((attack_ms / 500.0) * height))
    release_height = min(height, int((release_ms / 2000.0) * height))

    # Draw bars
    pygame.draw.rect(surface, (255, 200, 0), (x + width - 40, y + height - attack_height, bar_width, attack_height))
    pygame.draw.rect(surface, (0, 200, 255), (x + width - 20, y + height - release_height, bar_width, release_height))

    # === Labels ===
    labels = [
        f"Threshold: {threshold:.1f} dB",
        f"Ratio: {ratio:.1f}:1",
        f"Attack: {int(attack_ms)} ms",
        f"Release: {int(release_ms)} ms"
    ]
    for i, text in enumerate(labels):
        label = font.render(text, True, label_color)
        surface.blit(label, (x + 10, y + 10 + i * 18))

    # Label for bars
    atk_label = font.render("AK", True, label_color)
    rel_label = font.render("RL", True, label_color)
    surface.blit(atk_label, (x + width - 40, y + height + 4))
    surface.blit(rel_label, (x + width - 20, y + height + 4))


def draw_gain_threshold_visualizer(surface, x, y, width, height, params):
    """
    Visualizes gain and threshold levels in decibels.
    
    Parameters:
    - surface: pygame.Surface to draw on.
    - x, y: Top-left coordinates of the visualizer
    - width, height: Width and height of the visualizer
    - params: [gain_db (0 to 1), threshold_db (0 to 1)]
    """
    def lerp(t, a, b):
        return a + t * (b - a)
    
    raw_gain, raw_thresh = params
    gain_db = lerp(raw_gain, -60, 60)
    threshold_db = lerp(raw_thresh, -60, 0)

    # Background box
    pygame.draw.rect(surface, (30, 30, 45), (x, y, width, height), border_radius=10)

    # Meter bounds
    meter_x = x + width // 3
    meter_width = width // 3
    meter_top = y + 10
    meter_bottom = y + height - 10
    meter_height = meter_bottom - meter_top

    # Draw meter background
    pygame.draw.rect(surface, (10, 10, 20), (meter_x, meter_top, meter_width, meter_height))

    # Draw gain bar
    gain_y = meter_bottom - ((gain_db + 60) / 120) * meter_height
    bar_height = meter_bottom - gain_y
    if gain_db >= 15:
        gain_color = (((gain_db-15)*(17/3)), (-17/3 * gain_db) + 340, 0)
    else:
        gain_color = (0, 255, 0)

    pygame.draw.rect(surface, gain_color, (meter_x, gain_y, meter_width, bar_height))

    # Draw threshold line
    threshold_y = meter_bottom - ((threshold_db + 60) / 120) * meter_height
    pygame.draw.line(surface, (255, 100, 100), (meter_x, threshold_y), (meter_x + meter_width, threshold_y), 2)

    # Draw center 0 dB line
    zero_y = meter_bottom - ((0 + 60) / 120) * meter_height
    pygame.draw.line(surface, (180, 180, 180), (meter_x, zero_y), (meter_x + meter_width, zero_y), 1)

    # Labels
    font = pygame.font.SysFont("Arial", 14)
    label_color = (200, 200, 200)

    gain_label = font.render(f"Gain: {round(gain_db)} dB", True, label_color)
    thresh_label = font.render(f"Threshold: {round(threshold_db)} dB", True, label_color)

    surface.blit(gain_label, (x + 10, y + 10))
    surface.blit(thresh_label, (x + 10, y + 30))

    # dB scale labels (every 20 dB)
    for db in range(-60, 61, 20):
        label_y = meter_bottom - ((db + 60) / 120) * meter_height
        db_label = font.render(f"{db:+}", True, (140, 140, 140))
        surface.blit(db_label, (meter_x - 25, label_y - 7))


def draw_chorus_visualizer(surface, x, y, width, height, params, time=0):
    """
    Visualizes chorus effect with animated, parameter-driven graphics.
    
    Parameters:
        surface: Pygame surface to draw on.
        x, y: Top-left corner of visualizer.
        width, height: Dimensions.
        params (list): [rate_hz, depth, centre_delay_ms, feedback, mix]
        time (float): Time in seconds to animate wave.
    """
    rate_hz, depth, centre_delay_ms, feedback, mix = params

    # Interpolate actual values
    rate_hz = lerp(rate_hz, 0.1, 5)
    depth = lerp(depth, 0.0, 1.0)
    centre_delay_ms = lerp(centre_delay_ms, 1, 30)
    feedback = lerp(feedback, -0.95, 0.95)
    mix = lerp(mix, 0.0, 1.0)

    # Colors
    base_color = (0, 255, 0)
    feedback_color = (255,255,255)
    bg_color = (15, 30, 15)
    wave_color = (255, 255, 255)

    # Draw background
    pygame.draw.rect(surface, bg_color, (x, y, width, height), border_radius=16)

    # --- Wavy LFO Circle ---
    center = (x + width // 2, y + height // 2)
    radius = int(min(width, height) * 0.35)

    points = []
    for angle in range(0, 360, 5):
        rad = math.radians(angle)
        mod = math.sin(rad * rate_hz + time * 2 * math.pi * rate_hz)
        amp = radius + mod * radius * depth * 0.3
        px = center[0] + math.cos(rad) * amp
        py = center[1] + math.sin(rad) * amp
        points.append((px, py))

    pygame.draw.polygon(surface, wave_color, points, 2)

    # --- Delay Time Arc ---
    arc_radius = int(radius * 1.2)
    arc_thickness = int(centre_delay_ms / 2)
    start_angle = -math.pi / 2
    end_angle = start_angle + mix * 2 * math.pi

    pygame.draw.arc(surface, base_color,
                    (center[0] - arc_radius, center[1] - arc_radius, arc_radius * 2, arc_radius * 2),
                    start_angle, end_angle, arc_thickness)

    # --- Feedback Line ---
    fb_length = int(feedback * radius)
    pygame.draw.line(surface, feedback_color,
                     (center[0], center[1]),
                     (center[0] + fb_length, center[1]), 4)

    # --- Mix Fill Circle ---
    mix_radius = int(radius * 0.2 + mix * radius * 0.3)
    pygame.draw.circle(surface, "#67FF00", center, mix_radius)

