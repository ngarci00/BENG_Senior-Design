#Video decoding and encoding functions for ETI template
from __future__ import annotations
import os
from typing import Tuple
import torch
import torch.nn.functional as F
from torchvision.io import read_video #For reading videos using torchvision
    
def read_video_clip(video_path:str, num_frames:int, resize_hw: Tuple[int,int], random_clip: bool = True,) -> torch.Tensor:
    """Reads a video file and returns a tensor of shape (C, T, H, W) representing the video clip.
    
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract from the video.
        resize_hw (Tuple[int,int]): Height and width to resize the frames.
        random_clip (bool): Whether to extract a random clip or a center clip.
        
    Returns:
        torch.Tensor: Tensor of shape (C, T, H, W) representing the video clip.
  """ 
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    frames, _, _ = read_video(video_path, pts_unit='sec') #Read video using torchvision
    if frames.numel() == 0: #Check if frames were read successfully
        raise RuntimeError(f"Video decoding failed for file: {video_path}") # Raise error if decoding fails
    total = frames.shape[0]

    if total >= num_frames: #If video has enough frames
        start = (torch.randint(0, total - num_frames + 1, (1,)).item()
                 if random_clip else 0)
        frames = frames[start:start+num_frames] #Select frames
    else: #If video has fewer frames than required
        pad = num_frames - total
        frames = torch.cat([frames, frames[-1:].repeat(pad, 1, 1, 1)], dim=0) #Pad with last frame

    frames = frames.permute(0,3,1,2).contiguous().float() / 255.0 # Convert to [T, C, H, W] and normalize to [0, 1]
    frames = F.interpolate(frames, size=resize_hw, mode='bilinear', align_corners=False) #Resize frames
    return frames.permute(1,0,2,3).contiguous() #Return tensor in shape [C, T, H, W], contiguous makes sure the tensor is stored in a contiguous block of memory
