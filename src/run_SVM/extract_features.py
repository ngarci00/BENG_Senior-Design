import os, json, numpy as np, torch
from torch.utils.data import DataLoader

from src.run_2DCNN.dataset import VideoFrameDataset
from feature_extractor import ResNet18Embedder
import config

#Ensuring output directory exists
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def _video_to_embbeding(embedder: ResNet18Embedder, x: torch.Tensor) -> torch.Tensor:
    """Converts a batch of video frames to a single feature vector by averaging the frame-level features"""
    b, k, c, h, w = x.shape
    x2 = x.view(b*k,c,h,w) #Reshape to process all frames together
    z_frames = embedder(x2) #Extract features for all frames, shape [b*k, 512]
    z_frames = z_frames.view(b, k, -1) #Reshape back to [b, k, 512]
    z_video = z_frames.mean(dim=1) #Average the frame features to get a single feature vector per video, shape [b, 512]
    return z_video

def extract_fold(fold:int, device:str) -> None:
    ensure_dir(config.runs_dir) #Ensure the output directory exists

    output_path = os.path.join(config.features_dir, f"fold_{fold}.npz") #Path to save the extracted features for this fold
    if os.path.exists(output_path):
        print(f"Features for fold {fold} already exist at {output_path}, skipping extraction.")
        return
    
    #dataset loading for train and validation
    ds_train = VideoFrameDataset(fold=fold, split="train")
    ds_val = VideoFrameDataset(fold=fold, split="validation")

    #Dataloaders for train and validation
    dl_train = DataLoader(ds_train,batch_size=4,shuffle=False,num_workers=2, pin_memory=False)
    dl_val = DataLoader(ds_val,batch_size=4,shuffle=False,num_workers=2, pin_memory=False)

    embedder = ResNet18Embedder(pretrained=config.use_pretrained_backbone).to(device) #Initialize the ResNet-18 embedder and move it to the specified device
    embedder.eval() #Set the embedder to evaluation mode since we are only extracting features

    def run_loader(dl):
        X_list, y_list, vid_list = [], [], []
        with torch.no_grad(): #No need to compute gradients for feature extraction
            for x,y,video_id in dl:
                x = x.to(device) #Move the batch of video frames to the device
                z = _video_to_embbeding(embedder, x) #Extract features for the batch of videos
                X_list.append(z.cpu().numpy()) #Move the features back to CPU and store them
                y_list.append(y.cpu().numpy()) #Store the labels
                vid_list.extend(video_id) #Store the video ids
        X = np.concatenate(X_list, axis=0) #Concatenate the features for all
        y = np.concatenate(y_list, axis=0) #Concatenate the labels for all
        return X, y, np.array(vid_list) #Return the features, labels, and video ids as numpy arrays
    
    X_train, y_train, vids_train = run_loader(dl_train) #Extract features for the training set
    X_val, y_val, vids_val = run_loader(dl_val) #Extract features for

    np.savez_compressed(output_path, X_train=X_train, y_train=y_train, vids_train=vids_train, X_val=X_val, y_val=y_val, vids_val=vids_val) #Save the extracted features, labels, and video ids to a compressed .npz file
    print(f"Extracted features for fold {fold} saved to {output_path}")
    print(f"Train features shape: {X_train.shape}, Train labels shape: {y_train.shape}, Val features shape: {X_val.shape}, Val labels shape: {y_val.shape}")

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu" #Use MPS (Apple Silicon) if available, otherwise fall back to CPU
    print(f"Using device: {device}")
    for fold in range(config.kfolds):
        extract_fold(fold, device)

if __name__ == "__main__":
    main()