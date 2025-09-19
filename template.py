##for testing purpose only


from PIL import Image
import numpy as np

# -------------------
# STEP 1: Remove Green
# -------------------
# Open frame image
frame = Image.open("temp.png").convert("RGBA")

# Convert to numpy
data = np.array(frame)
r, g, b, a = data.T

# Define green tolerance
green_min = 100   # minimum "greenness"
red_max   = 120   # maximum red allowed
blue_max  = 120   # maximum blue allowed

# Mask: detect green screen
green_areas = (g > green_min) & (r < red_max) & (b < blue_max)

# Make green transparent
data[..., :-1][green_areas.T] = (0, 0, 0)
data[..., -1][green_areas.T] = 0

# Convert back to image
frame_transparent = Image.fromarray(data)

# -------------------
# STEP 2: Insert New Picture
# -------------------
# Open the picture you want to insert
background = Image.open("yourphoto.jpg").convert("RGBA")

# Resize background to match frame size (optional, adjust as needed)
background = background.resize(frame_transparent.size)

# Composite the two: background below, frame on top
combined = Image.alpha_composite(background, frame_transparent)

# Save the final image
combined.save("result.png")
