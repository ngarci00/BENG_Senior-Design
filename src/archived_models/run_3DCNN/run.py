import os, json, time, torch
from config import kfolds, runs_path, perfect_acc_tolerance
from train import train_3dcnn
from eval import evaluate_folds
from biomarker_eval import run_biomarker_eval
from utils import ensure_dir_exists


def _format_seconds(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {remaining_seconds:.2f}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:.2f}s"
    return f"{remaining_seconds:.2f}s"


def time_training(fold: int, device: str) -> tuple[dict, float]:
    """Train one fold and return its summary plus elapsed training time."""
    start = time.perf_counter()
    summary = train_3dcnn(fold, device)
    elapsed = time.perf_counter() - start
    print(f"Fold {fold}: 3DCNN training completed in {_format_seconds(elapsed)}")
    return summary, elapsed


def main():
    pipeline_start = time.perf_counter()

    ensure_dir_exists(runs_path)#ensure the directory for saving runs exists

    device = "mps" if torch.backends.mps.is_available() else "cuda" #MPS for Apple Silicon, otherwise cuda <- MPS : Metal Performance Shaders
    print(f"Using device: {device}") #print the device being used

    all_summaries = [] #initialize a list to store summaries for all folds
    training_times = []
    folds = list(range(int(kfolds)))

    for fold in folds: #iterate over the number of folds defined in the config
        summary, elapsed = time_training(fold, device) #train the 3D CNN for the current fold and get the summary of results
        all_summaries.append(summary) #append the summary to the list of all summaries
        training_times.append(elapsed)

        # if summary.get('best_val_accuracy', 0) >= 1.0 - perfect_acc_tolerance: #check if the best validation accuracy is close enough to perfect
        #     print(f"Early stopping at fold {fold} due to perfect validation accuracy.") #print a message indicating early stopping
        #     break #stop training further folds if perfect accuracy is achieved

    output_path = os.path.join(runs_path, "kfold_summary.json") #define the path to save the summary of results
    with open(output_path, 'w') as f: #open the file for writing
        json.dump(all_summaries, f, indent=2) #save the list of summaries as a json file 
    print(f"Saved k-fold summary to {output_path}") #print a message indicating where the summary has been saved

    evaluate_folds(folds=folds, device=device)
    for fold in folds:
        run_biomarker_eval(fold=fold)
    run_biomarker_eval()

    total_training_time = sum(training_times)
    print(f"Total 3DCNN training time: {_format_seconds(total_training_time)}")
    pipeline_elapsed = time.perf_counter() - pipeline_start
    print(f"Total 3DCNN pipeline runtime: {_format_seconds(pipeline_elapsed)}")


if __name__ == "__main__":
    main()
