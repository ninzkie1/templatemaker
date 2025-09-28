from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

@app.route("/process", methods=["POST"])
def process():
    # User uploaded photo
    user_photo = Image.open(request.files["frame"]).convert("RGBA")

    # Template with green screen area
    template = Image.open("temp3.png").convert("RGBA")

    data = np.array(template)
    r, g, b, a = data.T

    # Green screen detection
    green_min = 100
    red_max = 120
    blue_max = 120
    green_mask = (g > green_min) & (r < red_max) & (b < blue_max)

    # Create mask image (L mode)
    mask = Image.fromarray((green_mask.T * 255).astype(np.uint8), mode='L')

    # Resize user photo to template size
    user_photo_resized = user_photo.resize(template.size)

    # Create a blank RGBA canvas
    combined = template.copy()

    # Paste user photo only where the mask is white (green area)
    combined.paste(user_photo_resized, (0, 0), mask)

    img_io = io.BytesIO()
    combined.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png", as_attachment=True, download_name="result.png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
