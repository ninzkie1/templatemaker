from flask import Flask, request, send_file
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# Global caches
templates = {}
boxes_cache = {}
masks_cache = {}

@app.route("/")
def home():
    return "✅ App is running!", 200

@app.route("/ping")
def ping():
    return {"status": "ok"}, 200


def detect_magenta_boxes(template: Image.Image):
    """Detect bounding boxes of magenta areas in the template."""
    data = np.array(template.convert("RGB"))
    r, g, b = data[:, :, 0], data[:, :, 1], data[:, :, 2]

    # Detect magenta areas (looser threshold)
    magenta_mask = (r > 150) & (b > 150) & (g < 160)
    mask_array = (magenta_mask * 255).astype(np.uint8)
    mask = Image.fromarray(mask_array, mode="L")
    mask_np = np.array(mask)

    visited = np.zeros_like(mask_np, dtype=bool)
    boxes = []

    def flood_fill(x, y):
        stack = [(x, y)]
        min_x, max_x, min_y, max_y = x, x, y, y
        while stack:
            cx, cy = stack.pop()
            if (0 <= cx < mask_np.shape[1]) and (0 <= cy < mask_np.shape[0]) \
               and not visited[cy, cx] and mask_np[cy, cx] > 0:
                visited[cy, cx] = True
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
        return min_x, min_y, max_x+1, max_y+1

    for y in range(mask_np.shape[0]):
        for x in range(mask_np.shape[1]):
            if mask_np[y, x] > 0 and not visited[y, x]:
                boxes.append(flood_fill(x, y))

    # Keep only reasonably large boxes
    boxes = [b for b in boxes if (b[2] - b[0] > 40 and b[3] - b[1] > 40)]
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes, mask


def preload_all():
    """Preload templates and cache masks + boxes to speed up first request."""
    for name in ["magenta.png", "final3.png"]:
        template = Image.open(name).convert("RGBA")
        boxes, mask = detect_magenta_boxes(template)
        templates[name] = template
        boxes_cache[name] = boxes
        masks_cache[name] = mask
    print("✅ Templates preloaded and cached!")


def place_images(template_name: str, user_photos: list):
    """Place user photos into magenta boxes of template (using cached version)."""
    template = templates[template_name]
    boxes = boxes_cache[template_name]
    mask = masks_cache[template_name]

    combined = template.copy()

    if len(boxes) < len(user_photos):
        raise ValueError(f"Expected {len(user_photos)} magenta boxes, found {len(boxes)}")

    for i, (x1, y1, x2, y2) in enumerate(boxes[:len(user_photos)]):
        user_photo = user_photos[i].convert("RGBA")
        box_width, box_height = x2 - x1, y2 - y1
        user_fill = ImageOps.fit(user_photo, (box_width, box_height), method=Image.LANCZOS)
        mask_crop = mask.crop((x1, y1, x2, y2))
        combined.paste(user_fill, (x1, y1), mask_crop)

    return combined


def export_result(images, output_type: str):
    """Export result as PNG or PDF"""
    file_io = io.BytesIO()

    if output_type == "pdf":
        if isinstance(images, list):
            rgb_images = [img.convert("RGB") for img in images]
            rgb_images[0].save(
                file_io, "PDF", save_all=True, append_images=rgb_images[1:], resolution=100.0
            )
        else:
            images.convert("RGB").save(file_io, "PDF", resolution=100.0)
        file_io.seek(0)
        return send_file(
            file_io,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="result.pdf"
        )

    else:  # PNG
        if isinstance(images, list):
            images[0].save(file_io, "PNG")
        else:
            images.save(file_io, "PNG")
        file_io.seek(0)
        return send_file(
            file_io,
            mimetype="image/png",
            as_attachment=True,
            download_name="result.png"
        )


@app.route("/process", methods=["POST"])
def process_single():
    """Process 1 photo into magenta.png template"""
    if "frame1" not in request.files:
        return {"error": f"No frame1 provided. Got {list(request.files.keys())}"}, 400

    user_photo = Image.open(request.files["frame1"])
    combined = place_images("magenta.png", [user_photo])
    output_type = request.args.get("output", "png").lower()
    return export_result(combined, output_type)


@app.route("/process3shots", methods=["POST"])
def process_triple():
    """Process 3 photos into final3.png template"""
    user_photos = []
    for i in range(1, 4):
        file_key = f"frame{i}"
        if file_key not in request.files:
            return {"error": f"Missing {file_key}. Got {list(request.files.keys())}"}, 400
        user_photos.append(Image.open(request.files[file_key]))

    combined = place_images("final3.png", user_photos)
    output_type = request.args.get("output", "png").lower()

    return export_result(combined, output_type)


if __name__ == "__main__":
    preload_all()  # 🔥 Preload templates before first request
    app.run(host="0.0.0.0", port=5000)
