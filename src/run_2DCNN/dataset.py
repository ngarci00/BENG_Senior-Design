import os, json, random, torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from skimage.io import imread
from skimage.color import gray2rgb
from config import index_json_path, splits_json_path, resize_hw, use_only_annotated_frames, seed, frames_per_video_train, frames_per_video_validation, sample_mode_train, sample_mode_validation

img_extensions = [".jpg", ".jpeg", ".png"]

def sample_frame_indices(n: int,k: int, mode: str, rng: random.Random) -> List[int]:
    """Samples frame indices based on the specified mode."""
    if n <= 0:
        raise RuntimeError("No frames were found for this video!")
    if k <= 1:
        return [0]
    mode = (mode or "random").lower()
    if mode == "uniform":
        return [int(round(i*(n-1)/(k-1))) for i in range(k)]
    if n >= k:
        return rng.sample(range(n), k)
    return [rng.randrange(n) for _ in range(k)]

class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, fold: int, split: str, seed: int = seed):
        
        self.fold = int(fold)
        self.split = str(split)
        self.seed = int(seed)

        self.index = json.load(open(index_json_path))
        self.splits = json.load(open(splits_json_path))

        self.meta = {x["video_id"]: x for x in self.index}
        self.video_ids = list(self.splits[f"fold_{self.fold}"][self.split]) #Get the video ids for the specified fold and split

        if self.split.lower() == "train":
            self.k = int(frames_per_video_train)
            self.sample_mode = str(sample_mode_train)
        else:
            self.k = int(frames_per_video_validation)
            self.sample_mode = str(sample_mode_validation)
        #One item per video, aggregating frames internally:
    def __len__(self):
        return len(self.video_ids)
    def _get_frame_list(self, video_id: str) -> List[str]: #Gets the list of frame file names for the given video id, depending on whether we are using only annotated frames or all frames
        m = self.meta[video_id]
        if use_only_annotated_frames:
            frames = m.get("annotated_frame_names", [])
            #If there are no annotated frames, fall back to using all frames
            if len(frames) == 0: 
                frames = m.get("frames_names", [])
        else:
            frames = m.get("frames_names", [])
        return frames

    def __getitem__(self, idx: int):#Gets the clip and label for the given index
        video_id = self.video_ids[idx]
        m = self.meta[video_id]
        y = torch.tensor(int(m["label"]), dtype=torch.long)#Get the label for the video and convert it to a tensor

        rng = random.Random(self.seed+idx+1000*self.fold+idx)
        frames = self._get_frame_list(video_id)
        
        inds = sample_frame_indices(len(frames), self.k, self.sample_mode, rng) #Sample frame indices based on the specified mode
        frame_names = [frames[i] for i in inds] #Get the corresponding frame file names
        images: List[torch.Tensor] = []

        for frame in frame_names:
            path = os.path.join(m["frames_dir"], frame) #Get the full path to the frame image
            arr = imread(path) #Read the image as a numpy array
            if arr.ndim == 2: #If the image is grayscale, convert it to RGB by duplicating the single channel
                arr = gray2rgb(arr)
            elif arr.shape[2] == 4: #If the image has an alpha channel, remove it
                arr = arr[:,:,:3]
            im = torch.from_numpy(np.ascontiguousarray(arr)).permute(2,0,1).float() / 255.0 #Convert the image to a tensor of shape (C,H,W) and normalize to [0,1]
            images.append(im)

        x = torch.stack(images, dim=0) #Tensor of shape [K,C,H,W]
        if x.ndim == 4 and x.shape[0] in (1,3) and x.shape[1] == self.k:
            x = x.permute(1,0,2,3) #If the frames were read as (C,K,H,W), permute to (K,C,H,W)
        x = F.interpolate(x, size=resize_hw, mode="bilinear", align_corners=False) #Resize the frames to the specified size for model input
        if x.shape[1] != 3:
            raise RuntimeError(f"Expected 3 channels after laoding frames,but got x.shape={tuple(x.shape)}")
        return x.contiguous(), y, video_id #Return the clip tensor, label tensor, and video id as a string
    
#Create a dataset instance with the specified parameters
if __name__ == "__main__": #Test the dataset by creating an instance and getting a sample
    dataset = VideoFrameDataset(fold=0, split="train") 
    print(f"Dataset length (videos): {len(dataset)}")
    x, y, video_id = dataset[0]
    print(f"Sample clip shape: {tuple(x.shape)} (K,C,H,W), label: {float(y.item())}, video_id: {video_id}")
