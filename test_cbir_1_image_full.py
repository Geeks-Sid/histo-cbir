import glob
import os

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from hipt_4k import HIPT_4K
from hipt_heatmap_utils import *
from hipt_model_utils import eval_transforms as imported_eval_transforms
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
    coordinates = []

    if h < 4096 or w < 4096:
        padding = (
            (0, pad_h),
            (0, pad_w),
            (0, 0),
        )  # Define padding in the format ((top, bottom), (left, right), (0, 0))
        padded_img = np.pad(img, padding, mode="reflect")
        return np.expand_dims(padded_img, axis=0), [[0, 0]]

    else:
        patches = []
        for i in range(0, h, 4096):
            for j in range(0, w, 4096):
                if i + 4096 <= h and j + 4096 <= w:
                    patch = img[i : i + 4096, j : j + 4096]
                    patches.append(patch)
                    coordinates.append([i, j])

        return np.array(patches), coordinates


def evaluate_patches(patches, model):
    """Evaluates a list of patches and returns the outputs."""
    outputs = []
    transform = imported_eval_transforms()  # Initialize once if it's an object

    print("Evaluating patches...")

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


def process_single_image(input_path, output_path):
    # Get list of all image paths in the input folder
    image_path = os.path.join(input_path)

    # Load models once, outside the loop, to avoid redundant loading
    device = torch.device("cuda:0")
    model = load_models(device)

    # Create output file path
    output_file_name = os.path.basename(image_path).replace(".png", "_output.npy")
    output_file_path = os.path.join(output_path, output_file_name)

    # Process each image
    patches, coordinates = process_image(image_path)

    print(f"Number of patches: {len(patches)}")

    # print("Saving patches...")
    # # Save all patches as images with the correspoding coordinates in the filename in the output folder
    # for i, (patch, coord) in enumerate(zip(patches, coordinates)):
    #     patch_name = os.path.basename(image_path).replace(
    #         ".png", f"_patch_{i}_x{coord[0]}_y{coord[1]}.png"
    #     )
    #     patch_path = os.path.join(output_path, patch_name)
    #     io.imsave(patch_path, patch)

    # Evaluate patches
    outputs_4k_patch_size = evaluate_patches(patches, model)
    outputs_4k_patch_size = np.array(outputs_4k_patch_size).squeeze()

    # Save output
    np.save(output_file_path, outputs_4k_patch_size)

    # Save coordinates
    coordinates_file_name = os.path.basename(image_path).replace(
        ".png", "_coordinates.npy"
    )
    coordinates_file_path = os.path.join(output_path, coordinates_file_name)
    np.save(coordinates_file_path, np.array(coordinates))

    # Calculate metric and plot
    calculate_metric_and_plot(outputs_4k_patch_size)


# Function to calculate cosine similarity
def calculate_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Function to calculate Euclidean distance
def calculate_euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def calculate_metric_and_plot(output_all_patches):
    # Calculate cosine similarity
    cosine_similarity_matrix = squareform(pdist(output_all_patches, metric="cosine"))

    # Calculate Euclidean distance
    euclidean_distance_matrix = squareform(
        pdist(output_all_patches, metric="euclidean")
    )

    # Plot cosine similarity matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(cosine_similarity_matrix)
    plt.colorbar()
    # Save cosine similarity matrix
    plt.savefig(
        "/media/siddhesh/D1-k/Kaggle-Ovarian/output_folder_10077_2/cosine_similarity_matrix.png"
    )

    # Plot Euclidean distance matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(euclidean_distance_matrix)
    plt.colorbar()
    # Save cosine similarity matrix
    plt.savefig(
        "/media/siddhesh/D1-k/Kaggle-Ovarian/output_folder_10077_2/euc_distance_matrix.png"
    )


def analyze_query_image(query_path, db_path, coordinates_path):
    try:
        query_patch = io.imread(query_path)
        db = np.load(db_path)
        coordinates = np.load(coordinates_path)
    except Exception as e:
        print(f"Error: {e}")
        return None

    device = torch.device("cuda:0")
    model = load_models(device)
    transform = imported_eval_transforms()

    print("Evaluating Query Image...")
    transformed_patch = transform(query_patch)
    transformed_patch_tensor = torch.tensor(transformed_patch).unsqueeze(dim=0).float()

    with torch.no_grad():
        output = model(transformed_patch_tensor).cpu().numpy()

    cosine_similarity = []
    euclidean_distance = []
    for i in range(len(db)):
        cosine_similarity.append(calculate_cosine_similarity(output, db[i])[0])
        euclidean_distance.append(calculate_euclidean_distance(output, db[i]))

    cosine_similarity = np.array(cosine_similarity)
    euclidean_distance = np.array(euclidean_distance)

    top_10_cosine_similarity_index = np.argsort(cosine_similarity)[::-1][
        :10
    ]  # Descending order
    top_10_euclidean_distance_index = np.argsort(euclidean_distance)[
        :10
    ]  # Ascending order

    print("Cosine Similarity:", top_10_cosine_similarity_index)
    print("Top 10 most similar patches based on cosine similarity:")
    for i in top_10_cosine_similarity_index:
        print(coordinates[i])

    print("Top 10 least distance patches based on Euclidean distance:")
    for i in top_10_euclidean_distance_index:
        print(coordinates[i])

    return top_10_cosine_similarity_index, top_10_euclidean_distance_index


def plot_images(query_path, top_10_cosine_coordinates, output_path):
    # Create figure
    fig = plt.figure(figsize=(10, 20))

    # Set up a grid: one row for the query image, and a 2x5 grid for the top 10 images.
    gs = gridspec.GridSpec(3, 5, height_ratios=[1, 1, 1])

    # Plot the query image
    query_patch = io.imread(query_path)
    ax0 = plt.subplot(gs[0, :])
    ax0.imshow(query_patch)
    ax0.set_title("Query Image")
    ax0.axis("off")

    # Plot the 10 images with the highest cosine similarity from these coordinates
    for i, coordinates in enumerate(top_10_cosine_coordinates):
        row = (i // 5) + 1  # Determine the row, add 1 to account for query image row
        col = i % 5  # Determine the column
        image_path = (
            output_path + f"10077_patch_*_x{coordinates[0]}_y{coordinates[1]}.png"
        )
        image_path = glob.glob(image_path)[0]
        image = io.imread(image_path)
        ax = plt.subplot(gs[row, col])
        ax.imshow(image)
        ax.set_title(f"Top {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_path = "/media/siddhesh/D1-k/Kaggle-Ovarian/train_images/10077.png"
    output_path = "/media/siddhesh/D1-k/Kaggle-Ovarian/output_folder_10077_2/"
    query_path = output_path + "10077_patch_64_x20480_y16384.png"

    # Create the output folder if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # process_single_image(input_path=input_path, output_path=output_path)

    db_path = (
        "/media/siddhesh/D1-k/Kaggle-Ovarian/output_folder_10077_2/10077_output.npy"
    )
    coordinates_path = "/media/siddhesh/D1-k/Kaggle-Ovarian/output_folder_10077_2/10077_coordinates.npy"

    top_10_cosine_coordinates, top_10_euclidean_coordinates = analyze_query_image(
        query_path, db_path, coordinates_path
    )

    # Plot the 10 images with the highest cosine similarity from these coordinates
    # These images are already generated and named as 10077_patch_*_x<coord[0]>_y<coord[1]>.png

    # Plot the query image
    query_patch = io.imread(query_path)
    plt.imshow(query_patch)

    # Plot the 10 images with the highest cosine similarity from these coordinates
    for coordinates in top_10_cosine_coordinates:
        image_path = (
            output_path + f"10077_patch_*_x{coordinates[0]}_y{coordinates[1]}.png"
        )
        image_grep = glob.glob(image_path)[0]
        image = io.imread(image_grep)
        plt.imshow(image)

    # Call the function with the appropriate paths
    plot_images(query_path, top_10_cosine_coordinates, output_path)
