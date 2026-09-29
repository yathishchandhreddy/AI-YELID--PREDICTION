"""
generate_background.py - Generates a photorealistic high-resolution (1920x1080)
agricultural scene recreating the uploaded tractor spraying lush green crop field with golden morning sunlight.
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_field_image(width=1920, height=1080):
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)

    # 1. Base gradient: Rich agricultural field
    # Top-right is warm golden sunlight, bottom-left is deep lush green field
    arr = np.zeros((height, width, 3), dtype=np.float32)

    # Coordinates grid
    y_coords, x_coords = np.mgrid[0:height, 0:width]

    # Normalized coordinates
    u = x_coords / width
    v = y_coords / height

    # Diagonal perspective axis for crop rows (roughly from (-0.2, 1.2) to (0.8, -0.2))
    # Angle around 58 degrees
    angle = math.radians(58)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    diag_dist = (x_coords * cos_a + (height - y_coords) * sin_a)
    row_freq = 0.045  # Row frequency

    # Crop field vs plowed earth boundary line (angled across the right side)
    # The boundary runs roughly from (width * 0.75, 0) down to (width * 0.55, height)
    boundary_x = width * (0.62 + 0.18 * (1.0 - v) - 0.05 * np.sin(v * 4))
    is_crop = x_coords < boundary_x
    crop_factor = np.clip((boundary_x - x_coords) / 40.0, 0.0, 1.0)

    # Crop field textures
    # Primary green tones
    crop_base_r = 45 + 25 * v + 15 * np.sin(diag_dist * 0.01)
    crop_base_g = 115 + 40 * (1 - v * 0.4) + 15 * np.cos(diag_dist * 0.012)
    crop_base_b = 32 + 20 * v

    # Crop rows stripes (high-frequency furrow texture)
    row_waves = np.sin(diag_dist * row_freq)
    row_furrow = np.sin(diag_dist * row_freq * 2.0)
    crop_shading = 0.82 + 0.28 * (0.5 + 0.5 * row_waves) + 0.1 * (0.5 + 0.5 * row_furrow)

    # Deep tractor tire track lanes in the left portion of the field
    # Wheel track pair around u = 0.12 and u = 0.24 running parallel
    track_pos1 = (x_coords - (width * (0.09 + 0.14 * (1.0 - v))))
    track_pos2 = (x_coords - (width * (0.16 + 0.14 * (1.0 - v))))
    is_track1 = np.exp(-((track_pos1 / 18.0) ** 2))
    is_track2 = np.exp(-((track_pos2 / 18.0) ** 2))
    track_factor = np.clip(is_track1 + is_track2, 0.0, 1.0)

    # Brown tilled earth tones (right side of the field)
    earth_base_r = 110 + 35 * (1 - v * 0.3) + 15 * np.sin(x_coords * 0.02)
    earth_base_g = 82 + 25 * (1 - v * 0.3) + 10 * np.cos(y_coords * 0.02)
    earth_base_b = 55 + 20 * (1 - v * 0.3)
    earth_furrows = 0.88 + 0.22 * np.sin((x_coords * 0.8 + y_coords * 0.6) * 0.035)

    # Distant background fields (top 15% of the frame)
    horizon_factor = np.clip((height * 0.18 - y_coords) / (height * 0.18), 0.0, 1.0)
    horizon_r = 185 + 30 * u
    horizon_g = 175 + 25 * u
    horizon_b = 105 + 15 * u

    # Combine crop and earth
    r = (crop_base_r * crop_shading * (1.0 - track_factor * 0.45) + track_factor * 45) * crop_factor + (earth_base_r * earth_furrows) * (1.0 - crop_factor)
    g = (crop_base_g * crop_shading * (1.0 - track_factor * 0.55) + track_factor * 55) * crop_factor + (earth_base_g * earth_furrows) * (1.0 - crop_factor)
    b = (crop_base_b * crop_shading * (1.0 - track_factor * 0.35) + track_factor * 35) * crop_factor + (earth_base_b * earth_furrows) * (1.0 - crop_factor)

    # Blend distant horizon fields
    r = r * (1.0 - horizon_factor * 0.8) + horizon_r * (horizon_factor * 0.8)
    g = g * (1.0 - horizon_factor * 0.8) + horizon_g * (horizon_factor * 0.8)
    b = b * (1.0 - horizon_factor * 0.8) + horizon_b * (horizon_factor * 0.8)

    # Golden sunrise / morning light wash from upper-right
    sun_dist = np.sqrt(((x_coords - width * 0.95) / width) ** 2 + ((y_coords - height * 0.05) / height) ** 2)
    sun_glow = np.exp(-sun_dist * 1.8) * 0.42
    r += sun_glow * 180
    g += sun_glow * 140
    b += sun_glow * 45

    # Clip to valid RGB
    arr[:, :, 0] = np.clip(r, 0, 255)
    arr[:, :, 1] = np.clip(g, 0, 255)
    arr[:, :, 2] = np.clip(b, 0, 255)

    base_img = Image.fromarray(arr.astype(np.uint8), mode='RGB')

    # Now draw the tractor and spray boom in the right-center quadrant (approx x = width * 0.68, y = height * 0.44)
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    tx = int(width * 0.675)
    ty = int(height * 0.44)

    # 1. Spray mist cloud behind the boom
    # Mist expands backwards and downwards
    for i in range(240):
        # random particles along the boom
        boom_t = np.random.uniform(-0.55, 0.45)
        # boom angle matches the tractor orientation
        bx = tx + int(boom_t * 540)
        by = ty + int(boom_t * 180) + int(np.random.normal(35, 12))
        radius = int(np.random.uniform(15, 45))
        alpha = int(np.random.uniform(40, 110))
        ov_draw.ellipse(
            [bx - radius, by - radius, bx + radius, by + radius],
            fill=(235, 245, 240, alpha)
        )

    # Soften the mist overlay
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=8))
    ov_draw = ImageDraw.Draw(overlay)

    # 2. Tractor shadow
    ov_draw.ellipse([tx - 40, ty - 10, tx + 60, ty + 50], fill=(20, 30, 20, 140))
    ov_draw.ellipse([tx - 35, ty + 10, tx + 45, ty + 65], fill=(15, 25, 15, 120))

    # 3. Tractor Wheels (large black off-road agricultural tires)
    # Rear large wheels
    ov_draw.ellipse([tx - 32, ty - 22, tx - 12, ty + 24], fill=(30, 30, 32, 255))
    ov_draw.ellipse([tx + 28, ty - 2, tx + 48, ty + 44], fill=(30, 30, 32, 255))
    # Front smaller wheels
    ov_draw.ellipse([tx - 24, ty - 50, tx - 10, ty - 20], fill=(38, 38, 40, 255))
    ov_draw.ellipse([tx + 18, ty - 34, tx + 32, ty - 4], fill=(38, 38, 40, 255))

    # 4. Tractor Body (Bright Agricultural Red / Orange Cabin)
    # Engine hood (angled)
    ov_draw.polygon([
        (tx - 15, ty - 55),
        (tx + 15, ty - 45),
        (tx + 18, ty - 15),
        (tx - 12, ty - 15),
    ], fill=(215, 50, 35, 255))
    
    # Yellow / white front headlight glare
    ov_draw.ellipse([tx - 10, ty - 60, tx - 2, ty - 52], fill=(255, 245, 180, 220))
    ov_draw.ellipse([tx + 5, ty - 52, tx + 13, ty - 44], fill=(255, 245, 180, 220))

    # Glass Cabin (Dark tint with sun reflection)
    ov_draw.polygon([
        (tx - 18, ty - 20),
        (tx + 22, ty - 8),
        (tx + 20, ty + 18),
        (tx - 20, ty + 12),
    ], fill=(55, 65, 75, 240))
    # Cabin roof (white/grey)
    ov_draw.polygon([
        (tx - 20, ty - 22),
        (tx + 24, ty - 10),
        (tx + 22, ty - 2),
        (tx - 22, ty - 14),
    ], fill=(240, 242, 245, 255))

    # Rear Mounted Chemical Sprayer Tank (Orange/White tank)
    ov_draw.ellipse([tx - 15, ty + 14, tx + 25, ty + 48], fill=(235, 95, 30, 255))
    ov_draw.ellipse([tx - 8, ty + 20, tx + 18, ty + 42], fill=(245, 245, 245, 255))

    # 5. Wide Boom Sprayer Wings (Black truss metal booms spanning left and right)
    # Boom spans across crop rows
    boom_left_x = tx - 320
    boom_left_y = ty - 110
    boom_right_x = tx + 310
    boom_right_y = ty + 105

    # Main structural truss tube
    ov_draw.line([(boom_left_x, boom_left_y), (boom_right_x, boom_right_y)], fill=(35, 38, 42, 255), width=5)
    ov_draw.line([(boom_left_x, boom_left_y + 4), (boom_right_x, boom_right_y + 4)], fill=(80, 85, 90, 255), width=2)

    # Vertical truss struts along the boom
    for step in np.linspace(0.05, 0.95, 28):
        sx = int(boom_left_x + step * (boom_right_x - boom_left_x))
        sy = int(boom_left_y + step * (boom_right_y - boom_left_y))
        ov_draw.line([(sx, sy - 6), (sx, sy + 14)], fill=(45, 48, 52, 255), width=2)
        # Sprayer nozzle emitting fine white cone
        ov_draw.polygon([
            (sx - 1, sy + 12),
            (sx + 1, sy + 12),
            (sx + 8, sy + 45),
            (sx - 8, sy + 45)
        ], fill=(240, 250, 248, 70))

    # 6. Fine secondary spray mist highlights
    for _ in range(120):
        t = np.random.uniform(0.05, 0.95)
        sx = boom_left_x + t * (boom_right_x - boom_left_x) + np.random.normal(0, 10)
        sy = boom_left_y + t * (boom_right_y - boom_left_y) + np.random.uniform(15, 60)
        r_sz = np.random.uniform(4, 18)
        ov_draw.ellipse([sx - r_sz, sy - r_sz, sx + r_sz, sy + r_sz], fill=(255, 255, 255, int(np.random.uniform(50, 140))))

    # Composite the overlay onto the base image
    final_img = Image.alpha_composite(base_img.convert('RGBA'), overlay).convert('RGB')

    # Save to public and src directories
    import os
    os.makedirs('public', exist_ok=True)
    os.makedirs('src/assets', exist_ok=True)

    final_img.save('public/image.png', quality=95)
    final_img.save('public/farm_field.jpg', quality=95)
    final_img.save('src/assets/farm_field.jpg', quality=95)
    print("Agricultural background image successfully generated at public/image.png and src/assets/farm_field.jpg!")

if __name__ == '__main__':
    create_field_image()
