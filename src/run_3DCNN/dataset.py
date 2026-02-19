import os, json, random, torch
import torch.nn.functional as F
from torchvision.io import read_image
from config import index_json_path, splits_json_path, resize_hw, use_only_annotated_frames, seed

img_extensions = ['.jpg', '.jpeg', '.png']

class VideoClipDataset(torch.utils.data.Dataset):
    def __init__(self, fold, split, clip_len, clips_per_video,seed=seed):#Initializes the dataset with the specified parameters
        
        self.fold = fold
        self.split = split
        self.clip_len = int(clip_len)
        self.clips_per_video = int(clips_per_video)
        self.seed = seed

        self.index = json.load(open(index_json_path))
        self.splits = json.load(open(splits_json_path))

        self.meta = {x['video_id']: x for x in self.index}
        self.video_ids = list(self.splits[f"fold{fold}"][split])#Get the video ids for the specified fold and split

        #Each video repeats clips_per_video times in the dataset, so that we can sample multiple clips from each video:
        self.samples = []
        for video_id in self.video_ids:
            for k in range(self.clips_per_video):
                self.samples.append(video_id,k)
        
    def __len__(self): #The length is equal to the number of videos times the number of clips per video
        return len(self.samples)
    
    def _get_frame_list(self, video_id):#Gets the list of frames for a given video_id
        m = self.meta[video_id]
        if use_only_annotated_frames:
            frames = m.get('annotated_frames', [])
            if len(frames) == 0:#If there are no annotated frames, fall back to using all frames
                frames = m['frames']
        else:
            frames = m['frames']
        return frames
    
    def _sample_clip(self,frames,rng):#Samples a clip of len(clip_len) from the list of frames, using a random number generator rng
        T = self.clip_len
        n = len(frames)
        if n <= 0:
            raise RuntimeError("No frames were found for this video!")
        if n < T:
            #padding if there is not enough frames to sample a full clip
            idxs = list((range(n)) + [n-1]*(T-n))
            return [frames[i] for i in idxs]
        
        start = rng.randint(0, n - T)
        return frames[start:start+T]

    def __getitem__(self, idx):#Gets the clip and label for the given index
        video_id, k = self.samples[idx]
        m = self.meta[video_id]
        y = torch.tensor(int(m['label']), dtype=torch.float32)#Get the label for the video and convert it to a tensor

        rng = random.Random(self.seed+idx+1000*self.fold)
        frames = self._get_frame_list(video_id)
        clip_names = self._sample_clip(frames,rng)\
        
        #Load the frames & stack them into a tensor of shape (C,T,H,W):
        images = []
        for frame in clip_names:
            path = os.path.join(m["frames_dir"], frame)
            im = read_image(path).float() / 255.0 #Load the image and normalize it to [0,1]
            images.append(im)

        x = torch.stack(images, dim=1) #Stack the images into a tensor of shape (C,T,H,W)
        x = x.permute(1,0,2,3) #Permute to (T,C,H,W)

        #Resize the frames to the specified size:
        C, T, H, W = x.shape
        x2 = x.permute(1,0,2,3) #Permute to (C,T,H,W) for interpolation
        x2 = F.interpolate(x2, size=resize_hw, mode='bilinear', align_corners=False) #Resize the frames
        x2 = x2.permute(1,0,2,3).contiguous() #Permute back to (T,C,H,W) and make it contiguous in memory
        
        return x, y, video_id #Return the clip tensor, label, and video_id for the given index

