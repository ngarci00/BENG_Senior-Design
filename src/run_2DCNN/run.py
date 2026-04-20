import os, json, torch
from config import kfolds, runs_path, perfect_acc_tolerance
from train import train_2dcnn
from utils import ensure_dir_exists

def main():
    ensure_dir_exists(runs_path)#ensure the directory for saving runs exists

        #Prefer CUDA (NVIDIA) when available; otherwise use Apple MPS if available; else CPU.
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}") #print the device being used

    all_summaries = [] #initialize a list to store summaries for all folds

    for fold in range(kfolds): #iterate over the number of folds defined in the config
        summary = train_2dcnn(fold, device) #train the 2D CNN for the current fold and get the summary of results
        all_summaries.append(summary) #append the summary to the list of all summaries

    output_path = os.path.join(runs_path, "kfold_summary.json") #define the path to save the summary of results
    with open(output_path, 'w') as f: #open the file for writing
        json.dump(all_summaries, f, indent=2) #save the list of summaries as a json file 
    print(f"Saved k-fold summary to {output_path}") #print a message indicating where the summary has been saved

if __name__ == "__main__":
    main()