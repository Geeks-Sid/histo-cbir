from collections import Counter

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data(hdf5_path, csv_path):
    """
    Load data from an HDF5 file and a CSV file.

    Parameters:
    hdf5_path (str): Path to the HDF5 file.
    csv_path (str): Path to the CSV file.

    Returns:
    hdf5_file (HDF5 file object): Loaded HDF5 file.
    df (DataFrame): DataFrame loaded from the CSV file.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    # Open the HDF5 file and keep it open
    hdf5_file = h5py.File(hdf5_path, "r")
    return hdf5_file, df


def find_min_distances(query_patch, hdf5_file):
    """
    Find and return the minimum distances and corresponding indices between a query patch
    and all patches in the HDF5 file.

    Parameters:
    query_patch (numpy array): The query patch.
    hdf5_file (HDF5 file object): The HDF5 file containing patches.

    Returns:
    min_distances_dict (dict): Dictionary containing the minimum distance and index for each image.
    """
    # Initialize an empty dictionary to store minimum distances for each image
    min_distances_dict = {}

    # Loop through all image keys in the HDF5 file
    for key in hdf5_file["patches"].keys():
        # Initialize variables to store the minimum distance and corresponding index for each image
        min_distance_for_image = float("inf")
        min_index_for_image = None

        # Loop through each patch for the current image key
        for index, patch in enumerate(hdf5_file["patches"][key]):
            # Calculate the Euclidean distance between the query patch and the current patch
            distance = np.linalg.norm(query_patch - patch)

            # If this distance is smaller, update our minimum distance and index for this image
            if distance < min_distance_for_image:
                min_distance_for_image = distance
                min_index_for_image = index

        # Save the smallest distance and corresponding index for this image
        min_distances_dict[key] = (min_distance_for_image, min_index_for_image)

    return min_distances_dict


def get_most_common_label(sorted_items, df):
    """
    Obtain the most common label among the top images based on their minimum distances.

    Parameters:
    sorted_items (list): Sorted list of tuples containing image keys and their minimum distances and indices.
    df (DataFrame): DataFrame containing image labels.

    Returns:
    most_common_label (int or str): The most common label among the top images.
    """
    # Initialize an empty list to store labels of the top images
    labels = []

    # Loop through the sorted items and get labels for each from the DataFrame
    for key, (_, _) in sorted_items:
        image_label = df[df["image_id"] == int(key)].iloc[0, 1]
        labels.append(image_label)

    # Use Counter to find the most common label among these
    counter = Counter(labels)
    most_common_label, _ = counter.most_common(1)[0]

    return most_common_label


def main():
    # Paths to the HDF5 and CSV files
    hdf5_path = (
        "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/combined.hdf5"
    )
    csv_path = "/media/siddhesh/D2-K/Kaggle-Ovarian/train.csv"

    # Load both HDF5 and CSV data
    hdf5_file, df = load_data(hdf5_path, csv_path)

    # For demonstration, let's use an example image key
    image_key = "10077"
    image_data = hdf5_file["patches"][image_key]
    print(f"Image Key: {image_key}, Shape: {image_data.shape}")

    # Initialize a dictionary to keep track of label frequencies
    ans_dict = {}

    # Loop through each embedding in the image_data
    for embedding in tqdm(image_data):
        # Find minimum distances between this embedding and all patches in the dataset
        min_distances_dict = find_min_distances(embedding, hdf5_file)

        # Sort these minimum distances and pick the top 11
        sorted_items = sorted(min_distances_dict.items(), key=lambda x: x[1][0])[:11]

        # Find the most common label among these top 11
        most_common_label = get_most_common_label(sorted_items, df)

        # Update the frequency of this label in our answer dictionary
        ans_dict[most_common_label] = ans_dict.get(most_common_label, 0) + 1

    # The final label for the query image is the most frequent label among all embeddings
    ans = max(ans_dict, key=ans_dict.get)
    print(f"The label for the query image is: {ans}")


if __name__ == "__main__":
    main()
