#COnfiguration file for annotation driven model training and evaluation, defining paths, parameters, and settings used across the training and evaluation scripts.
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

index_json_path = os.path.join(repo_root, "data/videos/index_poly.json")
splits_json_path = os.path.join(repo_root, "data/videos/splits_poly_20.json")

kfolds = 4 #Number of folds for Cross-Validation

runs_dir = os.path.join(repo_root, "runs/biomarker_driven_model")
features_dir = os.path.join(runs_dir, "features")
models_dir = os.path.join(runs_dir, "models")
reports_dir = os.path.join(runs_dir, "reports")

biomarkers = [
    "vocal_cords",
    "arytenoids",
    "epiglottis",
    "esophagus",
    "endotracheal_tube"
]