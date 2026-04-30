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
