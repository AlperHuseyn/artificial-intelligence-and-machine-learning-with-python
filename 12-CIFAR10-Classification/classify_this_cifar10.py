"""
This module provides functionality to load and concatenate CIFAR-10 dataset files 
into a single NumPy array. It includes functions to unpickle CIFAR-10 data files and 
concatenate them, handling both training and test datasets.
"""

import os
import numpy as np
from typing import List, Dict, Any


def unpickle(file_path: str) -> Dict[bytes, Any]:
    """
    Unpickle a CIFAR dataset file.

    Args:
        file_path (str): The path to a CIFAR dataset file.

    Returns:
        Dict[bytes, Any]: The contents of the unpickled file.
    """
    import pickle

    with open(file_path, "rb") as fo:
        data_dict = pickle.load(fo, encoding="bytes")
    return data_dict


def concat_datasets(dataset_paths: List[str]) -> Dict[str, Any]:
    """
    Concatenates datasets from a list of CIFAR dataset file paths into a single NumPy array,
    combining both data and labels.

    Parameters:
        dataset_paths (List[str]): A list of paths to the CIFAR dataset files.

    Returns:
        np.ndarray: A NumPy array containing concatenated data and labels, where data has shape (N, 3072)
                    and labels are appended as the last column, resulting in shape (N, 3073) for the array.
    """

    data_arrays = []
    label_arrays = []

    for path in dataset_paths:
        dataset = unpickle(path)
        data_arrays.append(np.array(dataset[b"data"], dtype=np.int32))
        label_arrays.append(np.array(dataset[b"labels"], dtype=np.int32))

    combined_data = np.vstack(data_arrays)
    combined_labels = np.hstack(label_arrays)

    # Note that combined_labels need to be reshaped to be concatenated along the second axis
    combined_dataset = np.hstack((combined_data, combined_labels.reshape(-1, 1)))

    return combined_dataset


def main():
    # Define the directory containing CIFAR-10 batch files
    cifar_10_batches_dir = os.path.join(
        os.getcwd(), "12-CIFAR10-Classification", "cifar-10-batches-py"
    )

    # Build the training dataset paths
    batch_files = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    X_dataset_paths = [
        os.path.join(cifar_10_batches_dir, batch_file) for batch_file in batch_files
    ]

    # Build the test dataset path(s)
    y_dataset_paths = [os.path.join(cifar_10_batches_dir, "test_batch")]

    # Load and concatenate training and test datasets
    X_dataset = concat_datasets(X_dataset_paths)
    y_dataset = concat_datasets(y_dataset_paths)


if __name__ == "__main__":
    main()
