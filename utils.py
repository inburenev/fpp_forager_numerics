""" Utility functions """

import numpy as np


def load_metadata(filename):
    """
    Load the metadata from the simulation file, reading only the lines that start with '# '.

    Parameters
    ----------
    filename : str
        The path to the file containing the data.

    Returns
    -------
    metadata : dict
        A dictionary where keys are metadata names and values are the corresponding metadata values.
        - Numeric values are converted to floats.
        - The key 'observable' is stored as a string.

    Raises
    ------
    FileNotFoundError
        If the file specified by `filename` does not exist.
    ValueError
        If the file's metadata format is incorrect (e.g., missing ':' or invalid split).
    """
    metadata = {}

    try:
        with open(filename, 'r') as file:
            while True:
                line = file.readline().strip()
                if not line:  # End of file
                    break

                if line.startswith("# "):
                    key_value = line[2:].split(":", 1)
                    if len(key_value) != 2:
                        raise ValueError(f"Metadata line is not in expected format: {line}")

                    key, value = key_value
                    key = key.strip()
                    value = value.strip()

                    if key == 'observable':
                        metadata[key] = value
                    else:
                        metadata[key] = float(value)
                else:
                    # Stop reading when a non-metadata line is encountered
                    break

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        raise

    return metadata


def load_data(filename):
    """
    Load data from the file, including metadata and histogram data, and return as a single array.

    Parameters
    ----------
    filename : str
        The path to the file containing the data.

    Returns
    -------
    data : ndarray
        A 2D array with three columns:
        - x : The centers of the bins in the histogram.
        - counts : The number of samples in each bin.
        - l10_px : The log10 of the probability for each bin. Negative infinity values
          are assigned to bins where there are no counts.

    Raises
    ------
    KeyError
        If the expected metadata key (e.g., 'n_steps') is not found.
    ValueError
        If there is an error loading the numerical data.
    """
    # Load metadata
    metadata = load_metadata(filename)

    # Read numerical data, skipping rows corresponding to the metadata
    try:
        skip_rows = len(metadata)  # Number of metadata lines to skip
        numerical_data = np.loadtxt(filename, skiprows=skip_rows)
    except ValueError as e:
        print(f"Error loading numerical data: {e}")
        raise

    # Extract bin centers (x) and counts from the data
    x = numerical_data[:, 0]
    counts = numerical_data[:, 1]

    # Calculate the width of each bin
    bin_width = x[1] - x[0] if len(x) > 1 else 1  # Avoid division by zero if x has only one value

    # Initialize the log10 probability array with negative infinity
    l10_px = np.full(len(x), -np.inf)

    # Compute the log10 probability for bins with non-zero counts
    non_zero_counts = counts > 0
    if np.any(non_zero_counts):
        l10_px[non_zero_counts] = (
                np.log10(numerical_data[non_zero_counts, 2])  # Log10 of the 3rd column (probability)
                + np.log10(np.e) * (-numerical_data[non_zero_counts, 3])  # Adjust based on the 4th column (scaling)
                - np.log10(bin_width * metadata['n_steps'])  # Normalize by bin width and number of trajectories
        )

    # Combine x, counts, and l10_px into a single 2D array
    data = np.column_stack((x, counts, l10_px))

    return data