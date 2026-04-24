#!/usr/bin/env python3
import argparse
import glob
import math
import os
import re
from typing import Iterable, List

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a video from an image glob without relying on video_cli.")
    parser.add_argument("out_file", help="Output video or gif file")
    parser.add_argument("-i", "--input-files", required=True, help="Input glob, for example '*.jpg'")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--nframes", type=int, default=None)
    return parser.parse_args()


def natural_key(path: str) -> List[object]:
    parts = re.split(r"(\d+)", path)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def even(value: int) -> int:
    return int(math.ceil(value / 2.0) * 2)


def gather_files(pattern: str, nframes: int | None) -> List[str]:
    files = sorted(glob.glob(pattern), key=natural_key)
    if nframes is not None:
        files = files[: int(nframes)]
    if not files:
        raise ValueError(f"No files matched pattern: {pattern}")
    return files


def target_shape(files: Iterable[str]) -> tuple[int, int]:
    max_height = 0
    max_width = 0
    for path in files:
        with Image.open(path) as image:
            width, height = image.size
        max_height = max(max_height, int(height))
        max_width = max(max_width, int(width))
    return even(max_height), even(max_width)


def ensure_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.stack([frame, frame, frame], axis=-1)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return frame[:, :, :3]
    return frame


def center_pad(frame: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    frame = ensure_rgb(frame)
    height, width = frame.shape[:2]
    if height > target_height or width > target_width:
        raise ValueError(
            f"Frame shape {frame.shape[:2]} is larger than target shape {(target_height, target_width)}"
        )

    output = np.zeros((target_height, target_width, frame.shape[2]), dtype=frame.dtype)
    y0 = (target_height - height) // 2
    x0 = (target_width - width) // 2
    output[y0 : y0 + height, x0 : x0 + width] = frame
    return output


def main() -> int:
    args = parse_args()
    files = gather_files(args.input_files, args.nframes)
    height, width = target_shape(files)
    print(
        {
            "out_file": args.out_file,
            "input_files": args.input_files,
            "nframes": args.nframes,
            "fps": args.fps,
            "target_shape": (height, width),
            "matched_files": len(files),
        },
        flush=True,
    )

    writer = imageio.get_writer(
        args.out_file,
        fps=int(args.fps),
        macro_block_size=1,
        ffmpeg_log_level="error",
    )
    try:
        for idx, path in enumerate(files, start=1):
            frame = imageio.imread(path)
            frame = center_pad(frame, height, width)
            writer.append_data(frame)
            if idx % 50 == 0 or idx == len(files):
                print(f"Wrote {idx}/{len(files)} frames", flush=True)
    finally:
        writer.close()

    print(f"Wrote video to {os.path.abspath(args.out_file)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
