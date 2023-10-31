import argparse
import glob
import os
import threading
from queue import Queue

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
from skimage.transform import resize  # for upsampling
from torchvision import transforms
from tqdm import tqdm


def process_image(image_path, patch_size=2048, stride=1024, enable_resize=True):
    img = io.imread(image_path)
    if img is None:
        return "Invalid image path"

    # Resize the entire image to twice its size with interpolation
    img = cv2.resize(img, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    h, w, _ = img.shape
    pad_h, pad_w = 4096 - h, 4096 - w

    if h < 4096 or w < 4096:
        padding = ((0, pad_h), (0, pad_w), (0, 0))
        padded_img = np.pad(img, padding, mode="reflect")
        return np.expand_dims(padded_img, axis=0)
    else:
        patches = [
            img[i : i + patch_size, j : j + patch_size]
            for i in range(0, h, stride)
            for j in range(0, w, stride)
            if i + patch_size <= h and j + patch_size <= w
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


def process_all_images(
    input_folder,
    output_folder,
    queue_length=2,
    patch_size=2048,
    stride=1024,
    enable_resize=True,
):
    image_paths = glob.glob(os.path.join(input_folder, "*.png"))
    device = torch.device("cuda:0")
    model = load_models(device)

    patch_queue = Queue(maxsize=queue_length)

    def producer(image_paths):
        for image_path in image_paths:
            patches = process_image(image_path, patch_size, stride, enable_resize)
            patch_queue.put((image_path, patches))

    def consumer():
        while True:
            image_path, patches = patch_queue.get()
            if patches is None:
                break

            output_file_name = os.path.basename(image_path).replace(
                ".png", "_output.npy"
            )
            output_file_path = os.path.join(output_folder, output_file_name)

            if not os.path.exists(output_file_path):
                outputs_4k_patch_size = evaluate_patches(patches, model)
                np.save(output_file_path, outputs_4k_patch_size)

            patch_queue.task_done()

    producer_thread = threading.Thread(target=producer, args=(image_paths,))
    producer_thread.start()

    consumer_thread = threading.Thread(target=consumer)
    consumer_thread.start()

    producer_thread.join()
    patch_queue.put((None, None))  # Sentinel to stop consumer
    consumer_thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images in a folder.")

    parser.add_argument(
        "--input_folder",
        default="/media/siddhesh/D1-k/Kaggle-Ovarian/train_images/",
        help="Path to the folder containing input images.",
    )
    parser.add_argument(
        "--output_folder",
        default="/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/",
        help="Path to the folder where output will be saved.",
    )
    parser.add_argument(
        "--queue_length", type=int, default=2, help="Size of the pre-fetch queue."
    )
    parser.add_argument(
        "--patch_size", type=int, default=2048, help="Size of the image patches."
    )
    parser.add_argument(
        "--stride", type=int, default=2048, help="Stride for image patch extraction."
    )
    parser.add_argument(
        "--enable_resize",
        type=bool,
        default=True,
        help="Enable or disable resizing of patches to 4096x4096.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    process_all_images(
        args.input_folder,
        args.output_folder,
        args.queue_length,
        args.patch_size,
        args.stride,
        args.enable_resize,
    )
