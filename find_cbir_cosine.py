import csv

import h5py
import numpy as np


def get_query_patch_and_coordinates(
    patches_hdf5_file, coordinates_hdf5_file, patch_key
):
    """Get query patch data and its coordinates."""
    query_patch_data = patches_hdf5_file[patch_key][
        len(patches_hdf5_file[patch_key]) // 2
    ]
    patch_coordinates = coordinates_hdf5_file[patch_key][
        len(coordinates_hdf5_file[patch_key]) // 2
    ]
    return query_patch_data, patch_coordinates


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def find_most_similar_patch(
    query_patch_data, patches_hdf5_file, coordinates_hdf5_file, key
):
    """Find the patch that is most similar to the query_patch_data."""
    patches = np.array(patches_hdf5_file[key])
    query_patch_data_flatten = query_patch_data.flatten()
    patches_flatten = patches.reshape(patches.shape[0], -1)

    similarities = np.apply_along_axis(
        cosine_similarity, 1, patches_flatten, query_patch_data_flatten
    )
    max_similarity_index = np.argmax(similarities)

    return (
        similarities[max_similarity_index],
        coordinates_hdf5_file[key][max_similarity_index],
        max_similarity_index,
    )


def write_to_csv(data, file_name):
    """Write data to a CSV file."""
    with open(file_name, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["Image", "Max Similarity", "Coordinates", "Index", "Max Patches"]
        )
        for row in data:
            csv_writer.writerow(row)


if __name__ == "__main__":
    # Paths (ideally these should be command-line arguments or read from a config file)
    patches_hdf5_path = "patches.hdf5"
    coordinates_hdf5_path = "coordinates.hdf5"
    csv_file_name = "queried_coordinates.csv"

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
                max_similarity,
                most_similar_coordinates,
                most_similar_index,
            ) = find_most_similar_patch(
                query_patch_data, patches_hdf5_file, coordinates_hdf5_file, key
            )

            output_data.append(
                [
                    key,
                    max_similarity,
                    most_similar_coordinates,
                    most_similar_index,
                    len(coordinates_hdf5_file[key]),
                ]
            )

    write_to_csv(output_data, csv_file_name)
