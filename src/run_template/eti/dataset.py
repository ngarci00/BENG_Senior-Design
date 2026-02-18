#Json video dataset + sample template
import os,json,torch
from dataclasses import dataclass
from typing import Any, Dict, Optional
from torch.utils.data import Dataset
from .io_video import read_video_clip
from .config import DEFAULT_LABELS, dict_to_tensor

@dataclass # Decorator to automatically generate special methods like __init__()
class Sample: #Data structure for a single video sample
    video: torch.Tensor
    y_clip: torch.Tensor
    mask: Optional[torch.Tensor] #Mask for variable-length sequences
    meta: Dict[str,Any] #Metadata dictionary

class ETIDataset(Dataset):
    def __init__(self, json_path: str, split: Optional[str], num_frames: int, resize_hw: tuple, random_clip: bool):
        """Initializes the ETI video dataset.
        
        Args:
            json_path (str): Path to the JSON file containing dataset annotations.
            split (Optional[str]): Dataset split to use (e.g., 'train', 'val', 'test').
            num_frames (int): Number of frames to extract from each video.
            resize_hw (Tuple[int, int]): Height and width to resize video frames.
            random_clip (bool): Whether to extract random clips from videos.
        """
        with open(json_path, 'r') as f:
            manifest = json.load(f)

        self.samples = [
            s for s in manifest["samples"]
            if split is None or s.get("split") == split
        ]  # Filter samples by split

        self.num_frames = num_frames
        self.resize_hw = resize_hw
        self.random_clip = random_clip
    
    def __len__(self):
        return len(self.samples) #Return the number of samples in the dataset
    def __getitem__(self,idx):
        s = self.samples[idx] #Get the sample at the specified index

        video = read_video_clip(s["video_path"], self.num_frames, self.resize_hw, self.random_clip)
        y_clip = dict_to_tensor(s.get("label_presence", {}))
        return Sample(video=video, y_clip=y_clip, mask=None, meta=s)
    
def collate_fn(batch):
    """Collate function to combine multiple samples into a batch.
    
    Args:
        batch (List[Sample]): List of Sample objects.
        
    Returns:
        Sample: A single Sample object containing batched data.
    """
    videos = torch.stack([b.video for b in batch])
    labels = torch.stack([b.y_clip for b in batch])
    return videos, labels 
