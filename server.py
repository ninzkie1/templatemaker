from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

@app.route("/process", methods=["POST"])
def process():
    file = request.files["frame"]

    # Open Moana template
    frame = Image.open("temp3.png").convert("RGBA")

    data = np.array(frame)
    r, g, b, a = data.T

    # Detect green
    green_min = 100
    red_max   = 120
    blue_max  = 120
    green_areas = (g > green_min) & (r < red_max) & (b < blue_max)

    # Make green transparent
    data[..., :-1][green_areas.T] = (0, 0, 0)
    data[..., -1][green_areas.T] = 0
    frame_transparent = Image.fromarray(data)

    # User photo as background
    background = Image.open(file).convert("RGBA")
    background = background.resize(frame_transparent.size)

    # Merge
    combined = Image.alpha_composite(background, frame_transparent)

    img_io = io.BytesIO()
    combined.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png", as_attachment=True, download_name="result.png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
