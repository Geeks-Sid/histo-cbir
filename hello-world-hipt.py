import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from hipt_4k import HIPT_4K
from hipt_heatmap_utils import *
from hipt_model_utils import eval_transforms as imported_eval_transforms
from hipt_model_utils import get_vit4k, get_vit256
from scipy.spatial.distance import pdist, squareform
from skimage import io, util
from torchvision import transforms
from tqdm import tqdm


def process_image(image_path):
    """Reads and processes the image based on its dimensions."""
    img = io.imread(image_path)
    if img is None:
        return "Invalid image path"

    h, w, _ = img.shape
    pad_h, pad_w = 4096 - h, 4096 - w

    if h < 4096 or w < 4096:
        padding = (
            (0, pad_h),
            (0, pad_w),
            (0, 0),
        )  # Define padding in the format ((top, bottom), (left, right), (0, 0))
        padded_img = np.pad(img, padding, mode="reflect")
        return np.expand_dims(padded_img, axis=0)

    else:
        patches = [
            img[i : i + 4096, j : j + 4096]
            for i in range(0, h, 4096)
            for j in range(0, w, 4096)
            if i + 4096 <= h and j + 4096 <= w
        ]
        return np.array(patches)


def load_models(device):
    """Loads pretrained models and sets them to evaluation mode."""
    pretrained_weights256 = (
        "/home/siddhesh/Work/Projects/HIPT/HIPT_4K/Checkpoints/vit256_small_dino.pth"
    )
    pretrained_weights4k = (
        "/home/siddhesh/Work/Projects/HIPT/HIPT_4K/Checkpoints/vit4k_xs_dino.pth"
    )

    model = HIPT_4K(pretrained_weights256, pretrained_weights4k, device, device)
    model.eval()

    return model


def evaluate_patches(patches, model):
    """Evaluates a list of patches and returns the outputs."""
    outputs = []
    transform = imported_eval_transforms()  # Initialize once if it's an object

    for patch in tqdm(patches):
        transformed_patch = transform(patch)  # Use as an object

        # Convert to PyTorch tensor and adjust dimensions
        transformed_patch_tensor = (
            torch.tensor(transformed_patch).unsqueeze(dim=0).float()
        )

        with torch.no_grad():
            output = model(transformed_patch_tensor)
        outputs.append(output.cpu().numpy())

    return outputs


def process_all_images(input_folder, output_folder):
    # Get list of all image paths in the input folder
    image_paths = glob.glob(os.path.join(input_folder, "*.png"))

    # Load models once, outside the loop, to avoid redundant loading
    device = torch.device("cuda:0")
    model = load_models(device)

    for image_path in image_paths:
        # Create output file path
        output_file_name = os.path.basename(image_path).replace(".png", "_output.npy")
        output_file_path = os.path.join(output_folder, output_file_name)

        # Check if the image output is present in the toutput folder or not
        if not os.path.exists(output_file_path):
            # Process each image
            patches, coordinates = process_image(image_path)

            print(f"Number of patches: {len(patches)}")

            # Save all patches as images with the correspoding coordinates in the filename in the output folder
            for i, (patch, coord) in enumerate(zip(patches, coordinates)):
                patch_name = os.path.basename(image_path).replace(
                    ".png", f"_patch_{i}_x{coord[0]}_y{coord[1]}.png"
                )
                patch_path = os.path.join(output_folder, patch_name)
                io.imsave(patch_path, patch)

            # Evaluate patches
            outputs_4k_patch_size = evaluate_patches(patches, model)

            # Save output
            np.save(output_file_path, outputs_4k_patch_size)


if __name__ == "__main__":
    input_folder = "/media/siddhesh/D1-k/Kaggle-Ovarian/train_images/"
    output_folder = "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_all_images(input_folder, output_folder)
