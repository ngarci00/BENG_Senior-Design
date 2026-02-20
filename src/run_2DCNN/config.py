import os 

#Labels:
LABEL_TO_INT = {'FAIL': 0, 'PASS': 1} #As the name suggests, label to integer mapping
INT_TO_LABEL = {0: 'FAIL', 1: 'PASS'} #Integer to label mapping

#Data paths:
data_dir = os.path.join("data", "videos") #Base directory for video data
raw_frames_dir = os.path.join(data_dir, "raw_videos") #Directory for raw video frames
ann_dir = os.path.join(data_dir, "rectangle_label_videos") #Directory for annotation JSON files
# ann_dir = os.path.join(data_dir, "polygon_label_videos") #Directory for annotation JSON files #<- change to polygon_label_videos if using polygon annotations

index_json_path = os.path.join(data_dir, "index_rec.json") #Path to index JSON file for rectangle annotations
# index_json_path = os.path.join(data_dir, "index_poly.json") #Path to index JSON file for polygon annotations

# splits_json_path = os.path.join(data_dir, "splits_rec.json") #Path to splits JSON file for train/val splits (all videos)
splits_json_path = os.path.join(data_dir, "splits_rec_10.json") #<-- Temorary, smaller splits for quick testing (10)
# splits_json_path = os.path.join(data_dir, "splits_poly_10.json") #Path to splits JSON file for train/val splits

#Output directories:
runs_path = os.path.join("runs","run_3DCNN") #Base directory for training runs

#Model & Training Parameters:
kfolds = 3 #Number of folds for cross-validation
seed = 42 #Random seed for reproducibility
epochs = 10 #Number of training epochs try: 10,20, 50, 100 HERE <-
batch_size = 8 #Batch size for training 
learning_rate = 1e-4 #Learning rate for optimizer also refered to as alpha in some contexts (0.0001) <- Need to play with this value, 
#as 3D CNNs can be sensitive to learning rate. Start with a small value and adjust based on training stability and convergence.
weight_decay = 1e-2 #Weight decay for regularization

#Data Augmentation Parameters:
clip_len = 16 #Number of frames in each video clip sample, this should be < or = to the # of frames in the shortest video. 16 is common choice for 3D CNNs
clips_per_video_train = 5 #Number of training clips to sample per video per epoch. The more clips the longer the training time! <-
clips_per_video_val = 5 #Number of validation clips to sample per video per epoch. Fewer clips can speed up validation, but may give noisier estimates of val performance.

resize_hw = (112,112) #HxW to resize frames for model input. This is a common value for 3D CNNs
use_only_annotated_frames = True

#For early stopping (k-fold) and if acc reaches performance threshold:
stop_if_val_acc_perfect = True #Helps us save some time, stops once 100% val acc, set to False if you want to train for all epochs regardless of performance.
perfect_acc_tolerance = 1e-4 #Tolerance for considering validation acc as perfect, this way we avoid with floating points like 0.99999... being considered perfect.

#Model
use_pretrained_model = False #we aren't using a pretrained model in this case.

