import glob
import os

import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm

# Maximum allowed image size in pixels (adjust as needed)
PIL.Image.MAX_IMAGE_PIXELS = 21503024400


def get_image_dimensions(image_path):
    # Open the image file using PIL and get its dimensions
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def process_image(image_path, patch_size=4096, stride=4096, enable_resize=True):
    width, height = get_image_dimensions(image_path)

    if width < 4096 or height < 4096:
        return [(0, 0)]  # Return single patch with (0, 0) coordinates
    else:
        patch_coordinates = []  # Initialize list to store patch coordinates

        for i in range(0, height, stride):
            for j in range(0, width, stride):
                if i + patch_size <= height and j + patch_size <= width:
                    patch_coordinates.append((i, j))

        return patch_coordinates


def process_all_images(input_folder, output_folder):
    # Get list of all image paths in the input folder
    image_paths = glob.glob(os.path.join(input_folder, "*.png"))

    for image_path in image_paths:
        # Create output file path
        coordinates_file_name = os.path.basename(image_path).replace(
            ".png", "_coordinates.npy"
        )

        # Check if the image output is present in the output folder or not
        # Process each image
        coordinates = process_image(image_path)

        # Save coordinates
        np.save(
            os.path.join(output_folder, coordinates_file_name),
            np.array(coordinates),
        )
        print("Length of coordinates: ", len(coordinates), "for image: ", image_path)


if __name__ == "__main__":
    input_folder = "/media/siddhesh/D1-k/Kaggle-Ovarian/train_images/"
    output_folder = (
        "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder_with_coordinates/"
    )

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_all_images(input_folder, output_folder)
