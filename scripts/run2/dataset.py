# Goal here is to create a dataset class for loading index.json + splits.json, filters by k fold and returns 
# the tensors clips for the 3D CNN Model! Tensors are necessary as they are the input to the model.
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image

def load_image( path: Path, resize_hw: Tuple[int, int])-> torch.Tensor:
    """Loads an image from the given path and resizes it to an specified HxW."""
    img = Image.open(path).convert("RGB")
    if resize_hw is not None:
        img = img.resize((resize_hw[1],resize_hw[0]), resample = Image.BILINEAR)
    x = torch.from_numpy(__import__('numpy').array(img)).float() / 255.0  # Normalize to [0, 1]
    x = x.permute(2, 0, 1).contiguous() # Change to CxHxW
    return x

class VideoClipDataset(Dataset):
    """ Dataset for loading video clips based on index and splits JSON files, its job is to:
    READ: 
    - (videos+frames+annoated_frames+label) from index.json
    - (fold train/val video ids) from splits.json"""
    def __init__(self,index_json:str = "data/videos/index.json", splits_json:str = "data/videos/splits.json",
                 fold: int=0, split: str="train", clip_len: int=16, resize_hw: Tuple[int,int] = (112,112),
                 clips_per_video:int =20, only_annotated_frames: bool=False, seed: int=42) -> None:
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")

        self.index_json = Path(index_json)
        self.splits_json = Path(splits_json)
        self.fold = fold
        self.split = split
        self.clip_len = int(clip_len)
        self.resize_hw = resize_hw
        self.clips_per_video = int(clips_per_video)
        self.use_only_annotated_frames = bool(only_annotated_frames)
        self.seed = int(seed)

        items = json.loads(self.index_json.read_text())
        splits = json.loads(self.splits_json.read_text())

        fold_key = f"fold_{fold}"
        if fold_key not in splits:
            raise KeyError(f"{fold_key} not found in {self.splits_json}")

        allowed_ids = set(splits[fold_key][split])

        # Filter index items to those video_ids
        self.videos: List[Dict[str, Any]] = [x for x in items if x["video_id"] in allowed_ids]
        if len(self.videos) == 0:
            raise RuntimeError(f"No videos found for {fold_key}:{split}. Check splits.json and index.json.")

        # Precompute the effective frame lists we will sample from per video
        # If use_only_annotated_frames=True, we sample only from frames that have JSONs.
        for v in self.videos:
            frames = v.get("frames", [])
            ann_frames = v.get("annotated_frames", [])
            if self.use_only_annotated_frames:
                if len(ann_frames) == 0:
                    raise RuntimeError(f"Video {v['video_id']} has zero annotated_frames but use_only_annotated_frames=True.")
                v["_sample_frames"] = ann_frames
            else:
                if len(frames) == 0:
                    raise RuntimeError(f"Video {v['video_id']} has zero frames.")
                v["_sample_frames"] = frames

        # Deterministic epoch sizing: each video yields clips_per_video samples
        self._length = len(self.videos) * self.clips_per_video

    def __len__(self) -> int:
        return self._length

    def _rng_for_index(self, idx: int) -> random.Random:
        """
        Deterministic RNG per sample index (helps reproducibility across workers).
        """
        r = random.Random(self.seed + idx + 1000 * self.fold)
        return r

    def __getitem__(self, idx: int):
        # Map global idx -> (video_idx, clip_idx)
        video_idx = idx // self.clips_per_video
        clip_idx = idx % self.clips_per_video  # not used directly, but kept for clarity

        v = self.videos[video_idx]
        video_id = v["video_id"]
        y = int(v["label"])

        frames_dir = Path(v["frames_dir"])
        frame_names: List[str] = v["_sample_frames"]

        # Random temporal crop for train, deterministic-ish for val
        T = len(frame_names)
        if T == 0:
            raise RuntimeError(f"{video_id}: empty frame list after filtering.")

        if T >= self.clip_len:
            if self.split == "train":
                rng = self._rng_for_index(idx)
                start = rng.randint(0, T - self.clip_len)
            else:
                # center clip for validation (stable baseline)
                start = (T - self.clip_len) // 2
            chosen = frame_names[start : start + self.clip_len]
        else:
            # pad by repeating last available frame
            chosen = frame_names[:] + [frame_names[-1]] * (self.clip_len - T)

        # Load frames
        clip = []
        for fn in chosen:
            img_path = frames_dir / fn
            if not img_path.exists():
                raise FileNotFoundError(f"{video_id}: missing frame {img_path}")
            clip.append(load_image(img_path, self.resize_hw))

        # Stack to (T, C, H, W) -> (C, T, H, W)
        x = torch.stack(clip, dim=0).permute(1, 0, 2, 3).contiguous()

        return x, torch.tensor(y, dtype=torch.long), video_id
    
#Sanity Check! This should show us the shape of the tensor clips along with their labels and video ids. 
#Example: a tensor([1,1]) should have a matching video of (PASS, PASS) if both videos are labeled as 1.
#& a tensor([1,0]) should have a matching video of (PASS, FAIL) if one video is labeled as 1 and the other as 0 !
from torch.utils.data import DataLoader
from dataset import VideoClipDataset

ds = VideoClipDataset(fold=0, split="train", clip_len=16, resize_hw=(112,112), clips_per_video=2)
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

x, y, vid = next(iter(dl))
print(x.shape, y, vid)  # Expected output: torch.Size([2, 3, 16, 112, 112]) tensor([...]) ('video_id1', 'video_id2')