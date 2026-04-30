import os

from anatomy_tracking import config as tracking_config

ARCHIVE_ROOT = tracking_config.ARCHIVE_ROOT
PROJECT_ROOT = tracking_config.PROJECT_ROOT
REPO_ROOT = PROJECT_ROOT
DEFAULT_INDEX_JSON = tracking_config.DEFAULT_INDEX_JSON
DEFAULT_SPLITS_JSON = tracking_config.DEFAULT_SPLITS_JSON
DEFAULT_OUTPUT_DIR = os.path.join(ARCHIVE_ROOT, "outputs", "anatomy_detector")

TARGET_CLASSES = list(tracking_config.TARGET_CLASSES)
CLASS_TO_ID = {class_name: idx + 1 for idx, class_name in enumerate(TARGET_CLASSES)}
ID_TO_CLASS = {idx: class_name for class_name, idx in CLASS_TO_ID.items()}
BACKGROUND_CLASS_ID = 0
