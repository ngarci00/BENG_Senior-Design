import json
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import gray2rgb
from skimage.io import imread

from run_HYBRID.config import (
    frames_per_video_train,
    frames_per_video_validation,
    index_json_path,
    resize_hw,
    sample_mode_train,
    sample_mode_validation,
    seed,
    splits_json_path,
    use_only_annotated_frames,
)


def sample_frame_indices(n: int, k: int, mode: str, rng: random.Random) -> List[int]:
    if n <= 0:
        raise RuntimeError("No frames were found for this video.")
    if k <= 1:
        return [0]
    mode = (mode or "random").lower()
    if mode == "uniform":
        return [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
    if n >= k:
        return rng.sample(range(n), k)
    return [rng.randrange(n) for _ in range(k)]


class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, fold: int, split: str, seed_value: int = seed):
        self.fold = int(fold)
        self.split = str(split)
        self.seed = int(seed_value)

        with open(index_json_path, "r") as handle:
            self.index = json.load(handle)
        with open(splits_json_path, "r") as handle:
            self.splits = json.load(handle)

        self.meta = {item["video_id"]: item for item in self.index}
        self.video_ids = list(self.splits[f"fold_{self.fold}"][self.split])

        if self.split.lower() == "train":
            self.k = int(frames_per_video_train)
            self.sample_mode = str(sample_mode_train)
        else:
            self.k = int(frames_per_video_validation)
            self.sample_mode = str(sample_mode_validation)

    def __len__(self):
        return len(self.video_ids)

    def _get_frame_list(self, video_id: str) -> List[str]:
        meta = self.meta[video_id]
        if use_only_annotated_frames:
            frames = meta.get("annotated_frame_names", [])
            if not frames:
                frames = meta.get("frame_names") or meta.get("frames_names", [])
        else:
            frames = meta.get("frame_names") or meta.get("frames_names", [])
        return frames

    def __getitem__(self, idx: int):
        video_id = self.video_ids[idx]
        meta = self.meta[video_id]
        y = torch.tensor(int(meta["label"]), dtype=torch.long)

        rng = random.Random(self.seed + idx + 1000 * self.fold + idx)
        frames = self._get_frame_list(video_id)
        indices = sample_frame_indices(len(frames), self.k, self.sample_mode, rng)
        frame_names = [frames[index] for index in indices]

        images: List[torch.Tensor] = []
        for frame_name in frame_names:
            path = os.path.join(meta["frames_dir"], frame_name)
            arr = imread(path)
            if arr.ndim == 2:
                arr = gray2rgb(arr)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
            image = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
            images.append(image)

        x = torch.stack(images, dim=0)
        if x.ndim == 4 and x.shape[0] in (1, 3) and x.shape[1] == self.k:
            x = x.permute(1, 0, 2, 3)
        x = F.interpolate(x, size=resize_hw, mode="bilinear", align_corners=False)
        if x.shape[1] != 3:
            raise RuntimeError(f"Expected 3 channels after loading frames, but got x.shape={tuple(x.shape)}")
        return x.contiguous(), y, video_id
