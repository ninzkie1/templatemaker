from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np
import io

app = Flask(__name__)
CORS(app)

@app.route("/process", methods=["POST"])
def process():
    # User uploaded photo
    user_photo = Image.open(request.files["frame"]).convert("RGBA")

    # Template with magenta box area
    template = Image.open("magenta.png").convert("RGBA")

    # Convert to numpy to detect magenta
    data = np.array(template)
    r, g, b, a = data.T

    # Detect magenta (red + blue high, green low)
    red_min = 180
    blue_min = 180
    green_max = 100
    magenta_mask = (r > red_min) & (b > blue_min) & (g < green_max)

    # Make a mask image (L mode)
    mask_array = (magenta_mask.T * 255).astype(np.uint8)
    mask = Image.fromarray(mask_array, mode='L')

    # Get bounding box of magenta region automatically
    bbox = mask.getbbox()  # (x1, y1, x2, y2)

    combined = template.copy()  # start with template

    if bbox:
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1

        # Resize user photo to FILL the box (crop edges if needed)
        user_fill = ImageOps.fit(user_photo, (box_width, box_height), method=Image.LANCZOS)

        # Paste directly at top-left of the box
        overlay = Image.new("RGBA", template.size, (0, 0, 0, 0))
        overlay.paste(user_fill, (x1, y1))

        # Paste overlay into template using mask (only magenta area)
        combined.paste(overlay, (0, 0), mask)

    # Return image to user
    img_io = io.BytesIO()
    combined.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png", as_attachment=True, download_name="result.png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
