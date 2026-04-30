import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "videos")
JSON_DIR = os.path.join(DATA_DIR, "json_utils")

DEFAULT_INDEX_JSON = os.path.join(JSON_DIR, "index_poly.json")
# Main evaluation setting: use all 50 polygon-labeled videos for anatomy and fusion.
DEFAULT_SPLITS_JSON = os.path.join(JSON_DIR, "splits_poly_50.json")

DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")
TRACKING_FORMAT_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "anatomy_tracking_format")
TRACKS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "anatomy_tracks")
FEATURES_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "anatomy_features")
CLASSIFIER_RESULTS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "anatomy_classifier_results")
ENSEMBLE_RESULTS_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "ensemble_results")

#List of target anatomical classes for detection and tracking, which are the focus of the analysis and modeling efforts in the project.
TARGET_CLASSES = [
    "vocal_cords",
    "arytenoids",
    "epiglottis",
    "esophagus",
    "endotracheal_tube",
]

#Mapping of various label synonyms to standardized class names to ensure consistency in labeling and analysis across different videos and annotations.
LABEL_SYNONYMS = {
    "vocalcords": "vocal_cords",
    "vocalchords": "vocal_cords",
    "vocal cords": "vocal_cords",
    "vocal chords": "vocal_cords",
    "vocal_cord": "vocal_cords",
    "vocal_chords": "vocal_cords",
    "vocal_cords": "vocal_cords",
    "cords": "vocal_cords",
    "glottis": "vocal_cords",
    "epiglottis": "epiglottis",
    "epiglottitis": "epiglottis",
    "arytenoid": "arytenoids",
    "arytenoids": "arytenoids",
    "esophagus": "esophagus",
    "oesophagus": "esophagus",
    "ett": "endotracheal_tube",
    "et tube": "endotracheal_tube",
    "endotracheal tube": "endotracheal_tube",
    "endotracheal_tube": "endotracheal_tube",
    "tube": "endotracheal_tube",
}
