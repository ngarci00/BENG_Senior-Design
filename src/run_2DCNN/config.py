import os 

#Labels:
LABEL_TO_INT = {'FAIL': 0, 'PASS': 1} #As the name suggests, label to integer mapping
INT_TO_LABEL = {0: 'FAIL', 1: 'PASS'} #Integer to label mapping

#Data paths:
data_dir = os.path.join("data", "videos") #Base directory for video data
raw_frames_dir = os.path.join(data_dir, "raw_videos") #Directory for raw video frames
# ann_dir = os.path.join(data_dir, "rectangle_label_videos") #Directory for annotation JSON files
ann_dir = os.path.join(data_dir, "polygon_label_videos") #Directory for annotation JSON files #<- change to polygon_label_videos if using polygon annotations

# index_json_path = os.path.join(data_dir, "index_rec.json") #Path to index JSON file for rectangle annotations
index_json_path = os.path.join(data_dir, "index_poly.json") #Path to index JSON file for polygon annotations

# splits_json_path = os.path.join(data_dir, "splits_rec.json") #Path to splits JSON file for train/val splits (all videos)
# splits_json_path = os.path.join(data_dir, "splits_rec_20.json") #<-- Temporary, smaller splits for quick testing (10)
splits_json_path = os.path.join(data_dir, "splits_poly_20.json") #Path to splits JSON file for train/val splits

#Output directories:
runs_path = os.path.join("runs","run_2DCNN") #Base directory for training runs

#Model & Training Parameters:
kfolds = 4 #Number of folds for cross-validation
seed = 42 #Random seed for reproducibility
epochs = 20 #Number of training epochs try: 20, 50, 100 HERE <-
batch_size = 32 #Batch size for training <- 2D CNN is lighter than 3D CNN so we can afford a larger batch size
num_workers = 4 #Number of worker processes for data loading, adjust based system's capabilities
learning_rate = 1e-4 #Learning rate for optimizer also refered to as alpha in some contexts (0.0001) <- Need to play with this value, 
#as 3D CNNs can be sensitive to learning rate. Start with a small value and adjust based on training stability and convergence.
weight_decay = 1e-4 #Weight decay for regularization

#Frame Sampling / Augmentation:
frames_per_video_train = 16 # Number of frames to sample from each video for training
frames_per_video_validation = 32 # Frames to sample for validation.
sample_mode_train = "random" #Mode for sampling frames during training, options: "uniform", "random", "first_n"
sample_mode_validation = "uniform" #Mode for sampling frames during validation, options: "uniform", "random", "first_n"

resize_hw = (224,224) #ImageNet backbones typically use 224x224, we can also try (112x112) for a lighter load! which is what the 3d cnn uses
use_only_annotated_frames = True #Whether to use only annotated frames for train/val. 

#Model
backbone = "resnet18" #Backbone architecture for 2D CNN, options: "resnet18", "resnet34", "resnet50", etc. ResNet-18 is a good starting point for a balance of performance and speed.
use_pretrained_model = True #Reccommended baseline for 2D CNN, helps w/ convergence and performance

