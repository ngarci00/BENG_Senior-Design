#Config file for SVM model, contains all the parameters and paths needed for feature extraction and SVM training. 
import os

#data paths:
data_dir = os.path.join("data", "videos") #Base directory for video data
run_name = "run_SVM"
runs_dir = os.path.join("runs", run_name) #Directory to save SVM training runs, where run_name can be something like "svm_run_1"
features_dir = os.path.join(runs_dir, "svm_features") #Directory to save extracted features for
models_dir = os.path.join(runs_dir, "models") #Directory to save trained SVM models for each fold
reports_dir = os.path.join(runs_dir, "reports") #Directory to save training reports and metrics for each fold

# index_json_path = os.path.join(data_dir, "index_rec.json") #Path to index JSON file for rectangle annotations
index_json_path = os.path.join(data_dir, "index_poly.json") #Path to index JSON file for polygon annotations

# splits_json_path = os.path.join(data_dir, "splits_rec.json") #Path to splits JSON file for train/val splits (all videos)
# splits_json_path = os.path.join(data_dir, "splits_rec_20.json") #<-- Temporary, smaller splits for quick testing (10)
splits_json_path = os.path.join(data_dir, "splits_poly_50.json") #Path to splits JSON file for train/val splits

seed = 42 #Random seed
kfolds = 4 #Number of folds for cross-validation, should match the number of folds used in feature extraction 

#Originial video resolution is 1280x720
resize_hw = (600,600) #ImageNet backbones typically use 224x224, we can also try 112x112 for a lighter load!
#Need to test (64x64, 128x128, 224x224, 320x320, 600x600) for the best balance of speed and performance
use_only_annotated_frames = True #Whether to use only annotated frames for train/val, should match the setting used during feature extraction

frames_per_video_train = 16 #Number of frames to sample from each video for training, should match the setting used during feature extraction
frames_per_video_validation = 16 #Number of frames to sample from each video for validation, should match the setting used during feature extraction

sample_mode_train = "random" #Sampling mode for training frames
sample_mode_validation = "uniform" #Sampling mode for validation frames

#Feature extraction parameters:
num_workers = 0 #Number of DataLoader worker processes. Use 0 on macOS/sandboxed environments to avoid shared-memory worker errors
use_pretrained_backbone = True #Whether to use a pretrained ResNet-18 backbone for feature extraction
embedding_pool = "mean" #Pooling method to aggregate frame-level features into a video-level feature vector, can be "mean" or "max"

#SVM training parameters:
svm_kernel = "linear" #Kernel type for SVM, can be "rbf": Radial Basis Function, "poly": Polynomial, "sigmoid": Sigmoid, or "linear"
svm_C_grid = [0.1, 1.0, 10.0] #Regularization parameter for SVM, higher values mean less regularization
svm_n_jobs = 1 #Number of parallel jobs for GridSearchCV. Use 1 on macOS/sandboxed environments to avoid process-spawn errors
#if we use rbf then we need to specify gamma:
svm_gamma_grid = ["scale", "auto"] #Kernel coefficient for RBF, can be "scale" (1 / n_features) or "auto" (1 / n_features)
