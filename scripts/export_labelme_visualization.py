#!/usr/bin/env python3
import argparse
import base64
import io
import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple
from PIL import Image, ImageColor, ImageDraw

DEFAULT_COLORS = {
    "epiglottis": "#f97316",
    "endotracheal_tube": "#06b6d4",
    "vocal_cords": "#ef4444",
    "arytenoids": "#22c55e",
    "esophagus": "#a855f7",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render LabelMe JSON annotations into visualization frames.")
    parser.add_argument("input_path", help="LabelMe JSON file or directory containing JSON files")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to '<input>.export'",
    )
    parser.add_argument("--line-width", type=int, default=4)
    parser.add_argument("--label-box-height", type=int, default=24)
    parser.add_argument("--image-quality", type=int, default=95)
    return parser.parse_args()


def gather_json_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        if input_path.lower().endswith(".json"):
            return [input_path]
        raise ValueError(f"Expected a .json file, got: {input_path}")

    json_files: List[str] = []
    for root, _dirs, files in os.walk(input_path):
        for name in sorted(files):
            if name.lower().endswith(".json"):
                json_files.append(os.path.join(root, name))
    if not json_files:
        raise ValueError(f"No .json files found under {input_path}")
    return json_files


def default_output_dir(input_path: str) -> str:
    trimmed = input_path.rstrip(os.sep)
    return f"{trimmed}.export"


def relative_stem(base_dir: str, json_path: str) -> str:
    if os.path.isfile(base_dir):
        return os.path.splitext(os.path.basename(json_path))[0]
    rel = os.path.relpath(json_path, base_dir)
    return os.path.splitext(rel)[0]


def canonical_points(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    return [
        (float(point[0]), float(point[1]))
        for point in points
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]


def resolve_image(json_path: str, ann: Dict) -> Image.Image:
    image_data = ann.get("imageData")
    if image_data:
        blob = base64.b64decode(image_data)
        return Image.open(io.BytesIO(blob)).convert("RGB")

    image_path = ann.get("imagePath")
    if image_path:
        abs_path = os.path.normpath(os.path.join(os.path.dirname(json_path), str(image_path)))
        if os.path.exists(abs_path):
            return Image.open(abs_path).convert("RGB")

    fallback = os.path.splitext(json_path)[0] + ".jpg"
    if os.path.exists(fallback):
        return Image.open(fallback).convert("RGB")

    raise FileNotFoundError(f"Could not resolve source image for {json_path}")


def color_for_label(label: str) -> Tuple[int, int, int]:
    hex_color = DEFAULT_COLORS.get(label, "#e5e7eb")
    return ImageColor.getrgb(hex_color)


def box_from_points(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def draw_shape(draw: ImageDraw.ImageDraw, overlay: ImageDraw.ImageDraw, shape: Dict, label_box_height: int, line_width: int) -> None:
    label = str(shape.get("label") or "unknown")
    shape_type = str(shape.get("shape_type") or "polygon").lower()
    points = canonical_points(shape.get("points") or [])
    if not points:
        return

    color = color_for_label(label)
    fill = (*color, 56)

    if shape_type == "rectangle" and len(points) >= 2:
        x1, y1, x2, y2 = box_from_points(points[:2])
        overlay.rectangle((x1, y1, x2, y2), fill=fill)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        box = (x1, y1, x2, y2)
    else:
        if len(points) >= 3:
            overlay.polygon(points, fill=fill)
            draw.line(points + [points[0]], fill=color, width=line_width)
        else:
            draw.line(points, fill=color, width=line_width)
        box = box_from_points(points)

    x1, y1, _x2, _y2 = box
    label_y0 = max(0.0, y1 - float(label_box_height))
    label_y1 = label_y0 + float(label_box_height)
    text_width = max(60, 8 * len(label) + 12)
    draw.rectangle((x1, label_y0, x1 + text_width, label_y1), fill=color)
    draw.text((x1 + 6, label_y0 + 4), label, fill="white")


def render_visualization(json_path: str, line_width: int, label_box_height: int) -> Image.Image:
    with open(json_path, "r") as f:
        ann = json.load(f)

    image = resolve_image(json_path, ann)
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay, "RGBA")
    line_draw = ImageDraw.Draw(base)

    for shape in ann.get("shapes") or []:
        draw_shape(line_draw, overlay_draw, shape, label_box_height=label_box_height, line_width=line_width)

    composed = Image.alpha_composite(base, overlay)
    return composed.convert("RGB")


def export_frames(input_path: str, output_dir: str, line_width: int, label_box_height: int, image_quality: int) -> int:
    json_files = gather_json_files(input_path)
    total = 0
    for idx, json_path in enumerate(json_files, start=1):
        image = render_visualization(json_path, line_width=line_width, label_box_height=label_box_height)
        stem = relative_stem(input_path, json_path)
        frame_dir = os.path.join(output_dir, stem)
        os.makedirs(frame_dir, exist_ok=True)
        out_path = os.path.join(frame_dir, "visualization.jpg")
        image.save(out_path, quality=max(1, min(int(image_quality), 100)))
        total += 1
        if idx % 50 == 0 or idx == len(json_files):
            print(f"Exported {idx}/{len(json_files)} frames", flush=True)
    return total


def main() -> None:
    args = parse_args()
    input_path = os.path.abspath(args.input_path)
    output_dir = os.path.abspath(args.output_dir or default_output_dir(input_path))
    os.makedirs(output_dir, exist_ok=True)
    total = export_frames(
        input_path=input_path,
        output_dir=output_dir,
        line_width=args.line_width,
        label_box_height=args.label_box_height,
        image_quality=args.image_quality,
    )
    print(f"Wrote {total} visualization frames to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
