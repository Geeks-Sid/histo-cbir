import os

import h5py
import numpy as np
from tqdm import tqdm

# Specify directories containing the patch and coordinates files
patch_dir = "/media/siddhesh/D1-k/Kaggle-Ovarian/output_folder/"
coordinates_dir = (
    "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder_with_coordinates/"
)

# List all files in the directories
patch_files = os.listdir(patch_dir)
coordinates_files = os.listdir(coordinates_dir)

# Specify the path to the single HDF5 file where both patches and coordinates will be stored
hdf5_path = (
    "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/combined_new.hdf5"
)

# Create or open an HDF5 dataset
with h5py.File(hdf5_path, "w") as hdf5_file:
    # Create groups for patches and coordinates
    patches_group = hdf5_file.create_group("patches")
    coordinates_group = hdf5_file.create_group("coordinates")

    # Store patches
    for patch_file in tqdm(patch_files, desc="Processing patches"):
        patch_data = np.load(os.path.join(patch_dir, patch_file))
        key = os.path.splitext(patch_file)[0].replace("_output", "")
        patches_group[key] = patch_data

    # Store coordinates
    for coordinates_file in tqdm(coordinates_files, desc="Processing coordinates"):
        coordinates_data = np.load(os.path.join(coordinates_dir, coordinates_file))
        key = os.path.splitext(coordinates_file)[0].replace("_coordinates", "")
        coordinates_group[key] = coordinates_data

# Verify that the new HDF5 file was created successfully
with h5py.File(hdf5_path, "r") as output_hdf5_file:
    print("Groups in the output HDF5 file:", list(output_hdf5_file.keys()))

    # Check if lengths of coordinates and patches are the same for each image
    patches_group = output_hdf5_file["patches"]
    coordinates_group = output_hdf5_file["coordinates"]
    for key in patches_group.keys():
        print(
            "Length of coordinates and patches for image",
            key,
            "are",
            len(coordinates_group[key]),
            "and",
            len(patches_group[key]),
            "respectively",
        )
