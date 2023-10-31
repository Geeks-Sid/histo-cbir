from collections import Counter
from heapq import heappop, heappush

import h5py
import numpy as np
import pandas as pd

# Paths to HDF5 files
hdf5_path = "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/combined.hdf5"
df = pd.read_csv("/media/siddhesh/D2-K/Kaggle-Ovarian/train.csv")

# Open both HDF5 files in a single `with` statement
hdf5_file = h5py.File(hdf5_path, "r")

# Load patch data from the first HDF5 file
image_key = "10077"
image_data = hdf5_file["patches"][image_key]

print(image_key, image_data.shape)

# Select the corresponding coordinates from the second HDF5 file
all_coordinates_selected_patch = hdf5_file["coordinates"][image_key]

# get the coordinates of the patch
print(all_coordinates_selected_patch)
print(
    "Coordinates of the patch:",
    all_coordinates_selected_patch,
)


def get_patch_label(query_patch, db):
    # Initialize a dictionary to store the minimum distance and corresponding index for each image
    min_distances_dict = {}

    # Iterate over keys in the coordinates HDF5 file
    for key in hdf5_file["patches"].keys():
        # Initialize variables to keep track of the minimum distance and corresponding index for the current image
        min_distance_for_image = float("inf")
        min_index_for_image = None

        # Iterate over patches in the current selected key
        for index, patch in enumerate(hdf5_file["patches"][key]):
            # Compute the distance between query_patch_data and the patch
            distance = np.linalg.norm(query_patch - patch)

            # Update the minimum distance and index for the current image if this distance is smaller
            if distance < min_distance_for_image:
                min_distance_for_image = distance
                min_index_for_image = index

        # Store the minimum distance and index for the current image in the dictionary
        min_distances_dict[key] = (min_distance_for_image, min_index_for_image)

    # Select the top 10 unique images with the minimum distances
    sorted_items = sorted(min_distances_dict.items(), key=lambda x: x[1][0])[:11]

    # Initialize a list to store the labels for the top 10 images
    labels = []

    for key, (distance, index) in sorted_items:
        print("Reading image:", key)
        image_label = df[df["image_id"] == int(key)].values[0][
            1
        ]  # Extracting the first element
        print("Label:", image_label)
        labels.append(image_label)  # Append the scalar value, not the array

    # Use majority voting to determine the label for the query image
    counter = Counter(labels)
    most_common_label, _ = counter.most_common(1)[0]
    print(f"The most common label among the top 11 images is: {most_common_label}")

    return most_common_label


def get_image_label(image_embeddings, db):
    ans_dict = {}
    for embedding in image_embeddings:
        most_common_label = get_patch_label(embedding, db)
        ans_dict[most_common_label] = ans_dict.get(most_common_label, 0) + 1

    return max(ans_dict, key=ans_dict.get)


ans = get_image_label(image_data, hdf5_file)

print("The label for the query image is:", ans)
