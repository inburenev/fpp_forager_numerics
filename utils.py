""" Utility functions """
import numpy as np
import logging
import subprocess
from tqdm import tqdm


def create_configfile(filename: str, parameters: dict) -> None:
    """
    Creates a configuration file for the simulation program.

    This function takes a dictionary of parameters and writes them into a configuration
    file in the format required by an external C program used in simulation.
    The expected format and structure of the configuration file are detailed in the project's main README file.

    Parameters
    ----------
    filename : str
        The name of the configuration file to be created.
    parameters : dict
        A dictionary containing the simulation parameters, structured as follows:
            - 'model': dictionary of the parameters related to the model
            - 'simulation': dictionary of the parameters related to the simulation
            - 'result': dictionary of the parameters related to the resulting histogram

    Returns
    -------
    None

    """
    parameters_to_write = {
        'model': {
            'X0': 0,  'drift': 1, 'trajectory_length': 10000,
            'tau_distribution': f'exponential [{1}]',
            'M_distribution': f'exponential [{1}]',
        },
        'simulation': {
            'n_changes': -1, 'n_steps': 100,  'importance_sampling': 'none'
        },
        'result': {
            'observable': 'n',  'hist_min': 0, 'hist_max': 10000, 'n_bins': 1001
        }
    }

    for section in ['model', 'simulation', 'result']:
        if section in parameters.keys():
            for key in parameters_to_write[section].keys():
                if key in parameters[section].keys():
                    parameters_to_write[section][key] = parameters[section][key]

    with open(filename, 'w') as file:
        for section, params in parameters_to_write.items():
            file.write(f'[{section}]\n')
            for key, value in params.items():
                file.write(f'{key} = {value} \n')
            file.write('\n')

    return


def run_importance_sampling(theta_is_list: list[float], parameters: dict, n_changes_theta: dict) -> list[str]:
    """
    Perform importance sampling with an exponential tilt.

    This function performs importance sampling for a given list of quasi-temperatures (`theta_is_list`).
    The number of changes per metropolis step may vary depending on the quasi-temperature,
    it is passed as the dictionary 'n_changes_theta'.
    It writes simulation configuration files, executes an external simulation script program, and collects results.

    Parameters
    ----------
    theta_is_list : list[float]
        List of quasi-temperatures in the importance sampling scheme
    parameters : dict
        Dictionary of simulation parameters, including model, simulation, and result settings.
    n_changes_theta : dict
        Dictionary of the jumps that are subject to change for given quasi-temperature n_changes_theta[theta_is] = int

    Returns
    -------
    list[str]
        A list of filenames containing the results of the simulations.

    Notes
    -----
    - If `theta_is = 0`, no importance sampling is applied.
    - For acceptance rates below 0.7, a warning is logged
    - Simulations with overshoots stop further simulations for higher quasi-temperatures.
    """

    theta_is_max = np.max(theta_is_list)
    result_files = []
    n_changes_default = parameters['simulation']['n_changes']

    for theta_is in tqdm(theta_is_list):
        if theta_is <= theta_is_max:
            if theta_is == 0:
                parameters['simulation']['importance_sampling'] = 'none'
            else:
                parameters['simulation']['importance_sampling'] = f'exponential [{theta_is}]'

            parameters['simulation']['n_changes'] = n_changes_theta.get(theta_is, n_changes_default)

            create_configfile('conf.txt', parameters)
            fpp_output = subprocess.run(["./fpp", "conf.txt"], capture_output=True, text=True)
            result_file = (fpp_output.stdout.split("\n")[-1]
                           .split(":")[1].strip())
            # if the overshoot happens, we do not increase the temperature any further
            metadata = load_metadata(result_file)
            if int(metadata['overshoot']) > 0 and theta_is < theta_is_max:
                theta_is_max = theta_is

            result_files += [result_file]

            # low acceptance rate warning
            acc_rate = int(metadata['acc']) / int(metadata['n_steps'])
            if acc_rate < .7:
                logging.warning(f"Warning, low acceptance rate \n "
                                f"theta= {theta_is} acc_rate= {acc_rate} "
                                f"n_changes=, {n_changes_theta.get(theta_is, n_changes_default)}")

    # overshoot warning
    if theta_is_max < max(theta_is_list):
        logging.warning(f'Theta = {theta_is_max} led to overshoot. Simulations with higher quasi-temperatures skipped.')

    return result_files


def load_metadata(filename: str) -> dict:
    """
    Loads metadata from a simulation file, extracting lines starting with '# '.

    This function parses lines prefixed with `#` and extracts key-value pairs
    separated by a colon (`:`). The resulting metadata is returned as a dictionary.

    Parameters
    ----------
    filename : str
        The path to the input file

    Returns
    -------
    dict
        A dictionary of metadata
    """
    parameters = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                # Stop reading once we encounter a non-comment line
                if not line.startswith("#"):
                    break
                # Parse the parameter line if it starts with '#'
                parts = line[1:].split(":")
                if len(parts) == 2:
                    parameters[parts[0].strip()] = parts[1].strip()

    except FileNotFoundError:
        logging.error(f"File {filename} not found.")
    except Exception as e:
        logging.error(f"Error reading metadata from {filename}: {e}")
    return parameters


def glue_histograms(filenames: list[str], n_threshold: int) -> tuple[np.array, dict]:
    """
    Glues histograms from multiple files.

    Parameters
    ----------
    filenames : list of str
        List of filenames containing data.
    n_threshold : int
        Threshold for bin statistics. Bins with fewer than `n_threshold` elements are discarded.

    Returns
    -------
    tuple[np.array, dict]
        A tuple containing:
        - histogram (np.array): The combined histogram with 4 columns:
            - Column 0: Bin centers.
            - Column 1: Number of events in each bin.
            - Column 2, Column 3: Probability components to (split in two to avoid numerical overflow).
                            the probability is given by ( Column 2 ) * exp( - Column 3 )
        - ln_Z_diff (dict): Dictionary mapping `theta_is` values to logarithmic differences (ln Z_diff).

    Notes
    -----
    - The input files must have consistent binning.
    - The `theta_is = 0` (unbiased distribution) must be included in the input files.
    """

    filenames_dict = {float(load_metadata(filename)['theta_is']): filename for filename in set(filenames)}
    ln_Z_diff_dict = {}

    if 0 not in filenames_dict:
        raise ValueError("Unbiased distribution (theta_is = 0) not found.")

    # initialize the histogram with the unbiased distribution
    histogram = np.loadtxt(filenames_dict.pop(0))

    # Process each histogram in sorted order
    for theta_is in sorted(filenames_dict.keys(), key=abs):

        metadata_new = load_metadata(filenames_dict[theta_is])

        # skip simulations with overshoots -- they cannot be trusted
        if float(metadata_new['overshoot']) != 0:
            print(f" Overshoot detected in histogram with theta_is = {theta_is} ({metadata_new['observable']})."
                  f" Data will be ignored.")
            continue

        histogram_new = np.loadtxt(filenames_dict[theta_is])

        # the bin centers of the histograms should match
        if not np.array_equal(histogram[:, 0], histogram_new[:, 0]):
            raise ValueError(f"Bin centers for theta_is = {theta_is} do not match.")

        ind_overlap = (histogram[:, 1] > n_threshold) & (histogram_new[:, 1] > n_threshold)

        if ind_overlap.any():
            # compute the logarithmic difference between the probabilities
            ln_Z_diff = np.mean(
                np.log(histogram[ind_overlap, 2]) - histogram[ind_overlap, 3]
                - np.log(histogram_new[ind_overlap, 2]) + histogram_new[ind_overlap, 3]
            )
            ln_Z_diff_dict[theta_is] = ln_Z_diff

            better_stats = histogram_new[:, 1] > histogram[:, 1]
            histogram[better_stats, 1:] = histogram_new[better_stats, 1:]
            histogram[better_stats, 3] -= ln_Z_diff

    # the columns with bad statistics are fixed to 0
    histogram[histogram[:, 1] <= n_threshold, 1:] = 0

    return histogram, ln_Z_diff_dict
