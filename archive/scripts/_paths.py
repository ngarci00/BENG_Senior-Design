import os
import sys


ARCHIVE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(ARCHIVE_ROOT, ".."))
SRC_DIR = os.path.join(ARCHIVE_ROOT, "src")


def add_archive_src_to_path() -> None:
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)


def archive_path(*parts: str) -> str:
    return os.path.join(ARCHIVE_ROOT, *parts)


def project_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def _has_results(path: str) -> bool:
    if os.path.isfile(os.path.join(path, "all_folds_results.csv")):
        return True
    for fold in range(4):
        if os.path.isfile(os.path.join(path, f"fold_{fold}_results.csv")):
            return True
    return False


def default_hybrid_reports_dir() -> str:
    preferred = project_path("runs", "run_HYBRID", "reports")
    legacy_candidates = [
        project_path("runs", "run_SVM", "reports"),
        project_path("runs", "run_SVM", "res_eval", "reports_50Poly_224x224"),
    ]
    if _has_results(preferred):
        return preferred
    for candidate in legacy_candidates:
        if _has_results(candidate):
            return candidate
    return preferred


def default_hybrid_results_csv() -> str:
    return os.path.join(default_hybrid_reports_dir(), "all_folds_results.csv")
