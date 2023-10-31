import os

import h5py
import numpy as np

# Paths to HDF5 files
patches_hdf5_path = (
    "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/patches.hdf5"
)
coordinates_hdf5_path = "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/coordinates.hdf5"  # Replace with the actual path

# Create backup of the original files
patches_hdf5_backup_path = patches_hdf5_path + ".backup"
coordinates_hdf5_backup_path = coordinates_hdf5_path + ".backup"

# Create the copy
os.system(f"cp {patches_hdf5_path} {patches_hdf5_backup_path}")
os.system(f"cp {coordinates_hdf5_path} {coordinates_hdf5_backup_path}")


# Open the files and print the keys
with h5py.File(patches_hdf5_path, "r") as patches_hdf5_file, h5py.File(
    coordinates_hdf5_path, "r"
) as coordinates_hdf5_file:
    # Take the keyname in patches and remove the "_output" part
    patch_keyname = list(patches_hdf5_file.keys())[0]
    new_patch_keyname = patch_keyname.replace("_output", "")

    print("Keys in the patches HDF5 file:", list(patches_hdf5_file.keys()))
    print("Keys in the coordinates HDF5 file:", list(coordinates_hdf5_file.keys()))

    # Take the coordinates keyname and remove the "_coordinates" part
    coordinates_keyname = list(coordinates_hdf5_file.keys())[0]
    new_coordinates_keyname = coordinates_keyname.replace("_coordinates", "")

    # Check if the keyname and coordinates_keyname are the same
    if new_patch_keyname == new_coordinates_keyname:
        # Create new HDF5 files with updated keys
        with h5py.File(
            "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/updated_patches.hdf5",
            "w",
        ) as updated_patches_hdf5_file, h5py.File(
            "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/updated_coordinates.hdf5",
            "w",
        ) as updated_coordinates_hdf5_file:
            # Copy data from old files to new files with updated keys
            updated_patches_hdf5_file.create_dataset(
                new_patch_keyname, data=patches_hdf5_file[patch_keyname]
            )
            updated_coordinates_hdf5_file.create_dataset(
                new_coordinates_keyname, data=coordinates_hdf5_file[coordinates_keyname]
            )

# Read the HDF5 files and  Print the updated keys

with h5py.File(
    "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/updated_patches.hdf5",
    "r",
) as updated_patches_hdf5_file, h5py.File(
    "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/updated_coordinates.hdf5",
    "r",
) as updated_coordinates_hdf5_file:
    print(
        "Keys in the updated patches HDF5 file:", list(updated_patches_hdf5_file.keys())
    )
    print(
        "Keys in the updated coordinates HDF5 file:",
        list(updated_coordinates_hdf5_file.keys()),
    )
    print(updated_coordinates_hdf5_file.keys() == updated_patches_hdf5_file.keys())
