from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np
import io

app = Flask(__name__)
CORS(app)

@app.route("/process", methods=["POST"])
def process():
    shots = int(request.form.get("shots", 1))

    # choose template
    template_file = "magenta.png" if shots == 1 else "temp1.png"
    template = Image.open(template_file).convert("RGBA")
    combined = template.copy()

    # Convert to numpy to detect magenta boxes
    data = np.array(template)
    r, g, b, a = data.T
    red_min = 180
    blue_min = 180
    green_max = 100
    magenta_mask = (r > red_min) & (b > blue_min) & (g < green_max)
    mask_array = (magenta_mask.T * 255).astype(np.uint8)
    mask = Image.fromarray(mask_array, mode='L')

    # If multiple boxes in template, find each region
    # For simplicity, just take mask and split bounding boxes:
    from PIL import ImageChops
    # Find connected components:
    # (quick hack: for 1 box, just getbbox; for 3, manually crop)
    if shots == 1:
        frames = [request.files['frame1'] if 'frame1' in request.files else request.files['frame']]
        boxes = [mask.getbbox()]
    else:
        # user uploaded frame1, frame2, frame3
        frames = [request.files[f'frame{i+1}'] for i in range(3)]
        # For multiple boxes, you could predefine coordinates for each box
        # e.g. manually measure in your template:
        boxes = [(50,50,400,300),(450,50,700,300),(50,350,400,600)]  # example coords!

    overlay = Image.new("RGBA", template.size, (0,0,0,0))

    for frame_file, box in zip(frames, boxes):
        user_photo = Image.open(frame_file).convert("RGBA")
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        user_fill = ImageOps.fit(user_photo, (box_width, box_height), method=Image.LANCZOS)
        overlay.paste(user_fill, (x1, y1))

    combined.paste(overlay, (0,0), overlay)

    img_io = io.BytesIO()
    combined.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png", as_attachment=True, download_name="result.png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
