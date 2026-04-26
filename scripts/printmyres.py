import os
#Root directory of the repository
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_ROOT = os.path.join(REPO_ROOT, ".cache")
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

#Paths for the input image and output figure
IMAGE_PATH = os.path.join(REPO_ROOT, "data", "videos", "raw_videos", "41.MG", "00000207.jpg")
OUTPUT_PATH = os.path.join(REPO_ROOT, "outputs", "resolution_comparison.png")
ORIGINAL_OUTPUT_PATH = os.path.join(REPO_ROOT, "outputs", "resolution_original.png")
#List of resolutions to compare
RESOLUTIONS = [32, 64, 128, 224, 320, 600]

#ensure that the output directory exists
def ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

#load the image from the specified path
def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not read image: {path}")
    image = plt.imread(path)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image

#Display the original image in a figure with a title indicating its resolution
def build_original_figure(image: np.ndarray):
    src_h, src_w = image.shape[:2]
    figure, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    ax.imshow(image, interpolation="nearest")
    ax.set_title(f"Original ({src_w}x{src_h})", fontsize=18, fontweight="bold", pad=10)
    ax.tick_params(axis="both", labelsize=10, width=1.2, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    return figure

#resize the image to the specified side length while maintaining aspect ratio
def resize_image(image: np.ndarray, side: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    y_idx = np.linspace(0, src_h - 1, side).astype(np.int32)
    x_idx = np.linspace(0, src_w - 1, side).astype(np.int32)
    return image[np.ix_(y_idx, x_idx)]

#building a figure with subplots for each resolution just like the referenced paper:
def build_figure(image, resolutions):
    nrows, ncols = 2, 3
    figure, axes = plt.subplots(nrows, ncols, figsize=(16, 10), constrained_layout=True)

    for ax, side in zip(axes.ravel(), resolutions):
        resized = resize_image(image, side)
        ax.imshow(resized, interpolation="nearest")
        ax.set_title(f"{side}x{side}", fontsize=18, fontweight="bold", pad=10)
        ax.tick_params(axis="both", labelsize=10, width=1.2, length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    for ax in axes.ravel()[len(resolutions):]:
        ax.axis("off")

    return figure

#Creating a main function helps us organize the code and allows for better flow control.
def main() -> None:
    image = load_image(IMAGE_PATH)
    figure = build_figure(image, RESOLUTIONS)
    original_figure = build_original_figure(image)
    ensure_dir(OUTPUT_PATH)
    ensure_dir(ORIGINAL_OUTPUT_PATH)
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    original_figure.savefig(ORIGINAL_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    plt.close(original_figure)
    print(f"Saved resolution comparison to {OUTPUT_PATH}")
    print(f"Saved original image figure to {ORIGINAL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()