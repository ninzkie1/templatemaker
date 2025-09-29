from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np
import io

app = Flask(__name__)
CORS(app)


# Single-shot placement
def place_in_template(user_photo: Image.Image, template_path: str) -> Image.Image:
    template = Image.open(template_path).convert("RGBA")

    # Convert to numpy to detect magenta
    data = np.array(template)
    r, g, b, a = data.T

    # Detect magenta (red + blue high, green low)
    red_min = 180
    blue_min = 180
    green_max = 100
    magenta_mask = (r > red_min) & (b > blue_min) & (g < green_max)

    # Make a mask image
    mask_array = (magenta_mask.T * 255).astype(np.uint8)
    mask = Image.fromarray(mask_array, mode="L")

    # Get bounding box
    bbox = mask.getbbox()
    combined = template.copy()

    if bbox:
        x1, y1, x2, y2 = bbox
        box_width = x2 - x1
        box_height = y2 - y1

        # Resize photo
        user_fill = ImageOps.fit(user_photo, (box_width, box_height), method=Image.LANCZOS)

        overlay = Image.new("RGBA", (box_width, box_height), (0, 0, 0, 0))
        overlay.paste(user_fill, (0, 0))

        mask_crop = mask.crop((x1, y1, x2, y2))
        combined.paste(overlay, (x1, y1), mask_crop)

    return combined


@app.route("/process", methods=["POST"])
def process_single():
    if "frame1" not in request.files:
        return {"error": f"No frame1 provided. Got {list(request.files.keys())}"}, 400

    user_photo = Image.open(request.files["frame1"]).convert("RGBA")
    combined = place_in_template(user_photo, template_path="magenta.png")

    img_io = io.BytesIO()
    combined.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png", as_attachment=True, download_name="result.png")


@app.route("/process3shots", methods=["POST"])
def process_triple():
    template_path = "magenta3.png"   # template with 3 magenta boxes
    template = Image.open(template_path).convert("RGBA")
    combined = template.copy()

    # Detect magenta
    data = np.array(template)
    r, g, b, a = data.T
    red_min, blue_min, green_max = 180, 180, 100
    magenta_mask = (r > red_min) & (b > blue_min) & (g < green_max)

    mask_array = (magenta_mask.T * 255).astype(np.uint8)
    mask = Image.fromarray(mask_array, mode="L")
    mask_np = np.array(mask)

    # Flood fill bounding boxes
    visited = np.zeros_like(mask_np, dtype=bool)
    boxes = []

    def flood_fill(x, y):
        stack = [(x, y)]
        min_x, max_x, min_y, max_y = x, x, y, y
        while stack:
            cx, cy = stack.pop()
            if (0 <= cx < mask_np.shape[1]) and (0 <= cy < mask_np.shape[0]) and not visited[cy, cx] and mask_np[cy, cx] > 0:
                visited[cy, cx] = True
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
        return min_x, min_y, max_x+1, max_y+1

    for y in range(mask_np.shape[0]):
        for x in range(mask_np.shape[1]):
            if mask_np[y, x] > 0 and not visited[y, x]:
                boxes.append(flood_fill(x, y))

    # ✅ Sort correctly: top → bottom, then left → right
    boxes.sort(key=lambda b: (b[1], b[0]))

    if len(boxes) < 3:
        return {"error": f"Expected 3 magenta boxes, found {len(boxes)}"}, 400

    # Place each photo
    for i, (x1, y1, x2, y2) in enumerate(boxes[:3]):
        file_key = f"frame{i+1}"
        if file_key not in request.files:
            return {"error": f"Missing {file_key}. Got {list(request.files.keys())}"}, 400

        user_photo = Image.open(request.files[file_key]).convert("RGBA")
        box_width, box_height = x2 - x1, y2 - y1

        user_fill = ImageOps.fit(user_photo, (box_width, box_height), method=Image.LANCZOS)

        overlay = Image.new("RGBA", (box_width, box_height), (0, 0, 0, 0))
        overlay.paste(user_fill, (0, 0))

        mask_crop = mask.crop((x1, y1, x2, y2))
        combined.paste(overlay, (x1, y1), mask_crop)

    img_io = io.BytesIO()
    combined.save(img_io, "PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png", as_attachment=True, download_name="result.png")



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
