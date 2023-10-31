import csv

import h5py
import numpy as np


def get_query_patch_and_coordinates(
    patches_hdf5_file, coordinates_hdf5_file, patch_key
):
    query_patch_data = patches_hdf5_file[patch_key][
        len(patches_hdf5_file[patch_key]) // 2
    ]
    all_coordinates_selected_patch = coordinates_hdf5_file[patch_key]
    patch_coordinates = all_coordinates_selected_patch[
        len(all_coordinates_selected_patch) // 2
    ]
    return query_patch_data, patch_coordinates


def find_min_distance_patch(
    query_patch_data, patches_hdf5_file, coordinates_hdf5_file, key
):
    min_distance = float("inf")
    min_distance_coordinates = None
    min_distance_index = None

    for index, patch in enumerate(patches_hdf5_file[key]):
        distance = np.linalg.norm(query_patch_data - patch)
        if distance < min_distance:
            min_distance = distance
            min_distance_coordinates = coordinates_hdf5_file[key][index]
            min_distance_index = index

    return min_distance, min_distance_coordinates, min_distance_index


def write_to_csv(data, file_name):
    with open(file_name, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["Image", "Min Distance", "Coordinates", "Index", "Max Patches"]
        )
        for row in data:
            csv_writer.writerow(row)


if __name__ == "__main__":
    patches_hdf5_path = (
        "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/patches.hdf5"
    )
    coordinates_hdf5_path = (
        "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/coordinates.hdf5"
    )
    csv_file_name = "/media/siddhesh/D1-k/Kaggle-Ovarian/output_overlap_folder/queried_coordinates_temp.csv"

    output_data = []

    with h5py.File(patches_hdf5_path, "r") as patches_hdf5_file, h5py.File(
        coordinates_hdf5_path, "r"
    ) as coordinates_hdf5_file:
        patch_key = list(patches_hdf5_file.keys())[0]
        query_patch_data, patch_coordinates = get_query_patch_and_coordinates(
            patches_hdf5_file, coordinates_hdf5_file, patch_key
        )

        for key in coordinates_hdf5_file.keys():
            (
                min_distance,
                min_distance_coordinates,
                min_distance_index,
            ) = find_min_distance_patch(
                query_patch_data, patches_hdf5_file, coordinates_hdf5_file, key
            )

            output_data.append(
                [
                    key,
                    min_distance,
                    min_distance_coordinates,
                    min_distance_index,
                    len(coordinates_hdf5_file[key]),
                ]
            )

    write_to_csv(output_data, csv_file_name)
