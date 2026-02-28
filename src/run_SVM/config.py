#Config file for SVM model, contains all the parameters and paths needed for feature extraction and SVM training. 
import os

#data paths:
data_dir = os.path.join("data", "videos") #Base directory for video data
run_name = "run_SVM"
runs_dir = os.path.join("runs", run_name) #Directory to save SVM training runs, where run_name can be something like "svm_run_1"
features_dir = os.path.join(runs_dir, "svm_features") #Directory to save extracted features for
models_dir = os.path.join(runs_dir, "models") #Directory to save trained SVM models for each fold
reports_dir = os.path.join(runs_dir, "reports_10RecLabels") #Directory to save training reports and metrics for each fold

index_json_path = os.path.join(data_dir, "index_rec.json") #Path to index JSON file for rectangle annotations
# index_json_path = os.path.join(data_dir, "index_poly.json") #Path to index JSON file for polygon annotations

# splits_json_path = os.path.join(data_dir, "splits_rec.json") #Path to splits JSON file for train/val splits (all videos)
splits_json_path = os.path.join(data_dir, "splits_rec_10.json") #<-- Temporary, smaller splits for quick testing (10)
# splits_json_path = os.path.join(data_dir, "splits_poly_10.json") #Path to splits JSON file for train/val splits

seed = 42 #Random seed
kfolds = 3 #Number of folds for cross-validation, should match the number of folds used in feature extraction 

resize_hw = (224,224) #ImageNet backbones typically use 224x224, we can also try 112x112 for a lighter load!
use_only_annotated_frames = True #Whether to use only annotated frames for train/val, should match the setting used during feature extraction

frames_per_video_train = 16 #Number of frames to sample from each video for training, should match the setting used during feature extraction
frames_per_video_validation = 16 #Number of frames to sample from each video for validation, should match the setting used during feature extraction

sample_mode_train = "random" #Sampling mode for training frames
sample_mode_validation = "uniform" #Sampling mode for validation frames

#Feature extraction parameters:
use_pretrained_backbone = True #Whether to use a pretrained ResNet-18 backbone for feature extraction
embedding_pool = "mean" #Pooling method to aggregate frame-level features into a video-level feature vector, can be "mean" or "max"

#SVM training parameters:
svm_kernel = "linear" #Kernel type for SVM, can be "rbf": Radial Basis Function, "poly": Polynomial, "sigmoid": Sigmoid, or "linear"
svm_C_grid = [0.1, 1.0, 10.0] #Regularization parameter for SVM, higher values mean less regularization
#if we use rbf then we need to specify gamma:
svm_gamma_grid = ["scale", "auto"] #Kernel coefficient for RBF, can be "scale" (1 / n_features) or "auto" (1 / n_features)