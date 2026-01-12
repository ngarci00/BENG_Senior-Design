#Video decoding and encoding functions for ETI template
import os, torch
import decord #For video reading
import torch.nn.functional as F #For tensor operations
from typing import Tuple #For type hinting

def _try_import_decord():
    try:
        decord.bridge.set_bridge('torch') #Set decord to use PyTorch tensors
        return decord
    except Exception:
        return None
    
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
    decord = _try_import_decord()

    if decord is not None:
        vr = decord.VideoReader(video_path) #Use decord if available
        total = len(vr)

        if total >= num_frames: #If video has enough frames
            start = torch.randint(0, total - num_frames + 1, (1,)).item() if random_clip else 0
            idx = torch.arange(start, start + num_frames)
        else: #If video has fewer frames than required
            idx = torch.arange(total)
            idx = torch.cat([idx, idx[-1].repeat(num_frames - total)]) #Pad with last frame
                        #Padding refers to adding extra frames to reach the desired number of frames
        
        frames = vr.get_batch(idx) # [T, H, W, C] Selected frames
        frames = frames.permute(0,3,1,2).float() / 255.0 # Convert to [T, C, H, W] and normalize to [0, 1]
        #permute changes the order of dimensions in the tensor
        frames = F.interpolate(frames, size=resize_hw, mode='bilinear', align_corners=False) #Resize frames
        #interpolate resizes the frames to the specified height and width
        return frames.permute(1,0,2,3).contiguous() #Return tensor in shape [C, T, H, W], contiguous makes sure the tensor is stored in a contiguous block of memory
    from torchvision.io import read_video
    frames, _, _ = read_video(video_path, pts_unit='sec') #Read video using torchvision

    frames = frames.flaot() / 255.0 #Normalize to [0, 1]
    frames = frames.permute(0,3,1,2) #Convert to [T, C, H, W]
    frames = F.interpolate(frames, size=resize_hw, mode='bilinear', align_corners=False) #Resize frames
    return frames[:num_frames].permute(1,0,2,3) #Return tensor in shape [C, T, H, W]
