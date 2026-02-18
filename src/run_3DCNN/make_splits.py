import os, json
from sklearn.model_selection import StratifiedKFold
from config import index_json_path, splits_json_path, kfolds, seed

def main():
    with open(index_json_path, 'r') as f:
        items = json.load(f)

    #Create a list of video IDs and labels for stratified splitting    
    video_ids = [x['video_id'] for x in items]
    labels = [x['label'] for x in items]

    n_pos = sum(y == 1 for y in labels)
    n_neg = sum(y == 0 for y in labels)

    #Print the total # of videos & the distribution of PASS/FAIL labels
    print(f"Total videos: {len(video_ids)}, PASS: {n_pos}, FAIL: {n_neg}")

    #Check for class imbalance, if so then raises a warning:
    if n_pos != n_neg: 
        raise RuntimeError(f"Warning: The dataset is imbalanced, (PASS = {n_pos}, FAIL = {n_neg}). Please consider preprocessing of the data (: !")
    
    k = kfolds
    min_class = min(n_pos, n_neg)
    if k > min_class:
         k = min_class
    else:#If k< min_class: raise a ValueError, we can't have more folds than samples.
        raise ValueError(f"Error: Number of folds (k={k}) cannot be greater than number of samples in the smallest class (n={min_class}).")
    
    skf = StratifiedKFold(n_splits = k, shuffle=True, random_state=seed)

    splits = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(video_ids, labels)):
        splits[f'fold_{fold}'] = {
            'train': [video_ids[i] for i in train_idx],
            'val': [video_ids[i] for i in val_idx],
        }

        os.makedirs(os.path.dirname(splits_json_path), exist_ok=True)
        with open(splits_json_path, 'w') as f:
            json.dump(splits, f, indent=2) #saves the splits dictionary as a JSON file in the specified splits directory
        
        print(f"Wrote {k}-fold splits to {splits_json_path}!!!")

if __name__ == "__main__":
    main()