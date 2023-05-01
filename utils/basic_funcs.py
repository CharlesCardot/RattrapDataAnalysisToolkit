"""Useful functions for Rat Trap data processing."""
import os
import numpy as np
import pandas as pd

from scipy import integrate
from scipy import optimize

def all_runs_to_df(run_path):
    """
    Used to create a pandas dataframe from
    all the 'alldata' files in a directory.
    """

    alldata_files = [f.path for f in os.scandir(run_path)
                     if "alldata" in str(f.path)]

    for key, line in enumerate(open(alldata_files[0]).readlines()):
        if line.startswith("***"):
            rows_to_skip = key + 1

    data = [pd.read_csv(x, delim_whitespace=True, skiprows=rows_to_skip)
            for x in alldata_files]
    return data

def get_spectra_arr(run_path, runs):
    """
    Return spectra belonging to a given directory
    in list format, where every entry is an individual 
    spectra.

    Read in an 'alldata' file, taking either a
    list of runs or 'all', and returns a numpy array of the form
    [[x_values, y_values], [x_values, y_values], [...], ...].

    parameters:
        run_path - local path/ directory to data files
        runs - either 'all' or a list of run/scan numbers

    returns:
        array - [[x_values, y_values], [x_values, y_values], [...], ...]
    """
    data = all_runs_to_df(run_path=run_path)
    x = data[0]["Energy_(eV)"]

    if isinstance(runs, str) and runs == "all":
        # Return all runs
        spectra_arr = np.asarray([[x, d["cnts_per_live"]] for d in data])
        return spectra_arr

    elif isinstance(runs, list):
        # Return the runs specified in the list 'runs'
        spectra_arr = np.asarray([[x, d["cnts_per_live"]] for key, d
                                      in enumerate(data) if key in runs])
        return spectra_arr

    else:
        raise ValueError("runs must be either string 'all' or list " +
                         "(ex:[0, 1, 2]) denoting the which specific " +
                         "runs you wish to sum together")

def get_spectra_summed(run_path, runs):
    """
    Return spectra belonging to a given directory.

    Read in an 'alldata' file, sum together runs (either a
    list of runs or 'all'), and return a numpy array of the form
    [[x_values], [y_values]].

    parameters:
        run_path - local path/ directory to data files
        runs - either 'all' or a list of batch numbers

    returns:
        spectra - array in the form [energy, spectral sum of all batches]
    """
    data = all_runs_to_df(run_path=run_path)
    x = data[0]["Energy_(eV)"]

    if isinstance(runs, str) and runs == "all":
        # Return a sum of all runs
        y = np.asarray([d["cnts_per_live"] for d in data])
        return np.asarray([x, np.sum(y, axis=0)])

    elif isinstance(runs, list):
        # Return a sum of the runs specified in the list 'runs'
        y = np.asarray([dat["cnts_per_live"] for key, dat in enumerate(data)
                        if key in runs])
        return np.asarray([x, np.sum(y, axis=0)])

    else:
        raise ValueError("runs must be either string 'all' or list " +
                         "(ex:[0, 1, 2]) denoting the which specific " +
                         "runs you wish to sum together")


def find_index_of_closest_value(array, val):
    """Project value onto index."""
    dif = (array - val)**2
    return np.argmin(dif)


def subtract_constant_background(arr, roi=None):
    """
    Remove constanct background.

    Subtract a constant background from the x-y spectra using
    the intensity within the region of interest (ROI).

    parameters -
        arr - tuple or list in the form (energy, spectrum)
        roi - tuple in the form (start energy, end energy)
            default = None
    output -
        normalized_arr - tuple in the form (energy, processed spectrum)
    """
    if roi is None:
        y = arr[1]
    else:
        start_energy, end_energy = roi
        start_index = find_index_of_closest_value(arr[0], start_energy)
        end_index = find_index_of_closest_value(arr[0], end_energy)
        y = arr[1][start_index:end_index]

    def loss(b, y):
        fit = b  # constant background
        return np.sqrt(np.sum((y - fit)**2 / len(y)))

    starting_param_vals = [0]  # starting guess of zero background

    optimized_background = optimize.minimize(loss, x0=starting_param_vals,
                                             args=(y), method='BFGS')
    background = optimized_background['x']

    return np.asarray([arr[0], arr[1] - background])


def subtract_linear_background(arr, left_roi=None, right_roi=None):
    """
    Remove linear background.

    Subtract linear background from the x-y spectra. Fit a line through
    both the left and right ROIs, if given. By default, the fit is
    through the entire spectrum.

    parameters -
        arr - tuple or list in the form (energy, spectrum)
        left_roi - tuple in the form (left start energy, left end energy)
            default = None
        right_roi - tuple in the form (right start energy, right end energy)
            default = None
    output -
        normalized_arr - tuple in the form (energy, processed spectrum)
    """
    if left_roi is None and right_roi is None:
        energy, y = arr
    else:
        y = []
        energy = []
        for roi in [left_roi, right_roi]:
            if roi is not None:
                start_energy, end_energy = roi
                start_index = find_index_of_closest_value(arr[0], start_energy)
                end_index = find_index_of_closest_value(arr[0], end_energy)
                y.append(arr[1][start_index: end_index])
                energy.append(arr[0][start_index: end_index])
        y = np.array(y).reshape(-1)
        energy = np.array(energy).reshape(-1)

    def loss(x, y, energy):
        m, b = x
        fit = m * energy + b  # linear background
        return np.sqrt(np.sum((y - fit)**2 / len(y)))

    starting_param_vals = [0, 0]  # starting guess of slope and y-intercept

    optimized_background = optimize.minimize(loss, x0=starting_param_vals,
                                             args=(y, energy), method='BFGS')
    m, b = optimized_background['x']
    background = m * arr[0] + b

    return np.asarray([arr[0], arr[1] - background])


def subtract_linear_background_avg(arr, left_roi, right_roi):
    """
    Remove linear background.

    Subtract linear background from the x-y spectra. Find
    average x and y values within the two ROIs and use the
    line connecting them.
    """
    left_start_energy, left_end_energy = left_roi
    right_start_energy, right_end_energy = right_roi

    left_chunk = plottrim(arr, left_start_energy, left_end_energy)
    right_chunk = plottrim(arr, right_start_energy, right_end_energy)

    left_avg = np.average(left_chunk, axis=1)  # [x_{1,avg}, y_{1,avg}]
    right_avg = np.average(right_chunk, axis=1)  # [x_{2,avg}, y_{2,avg}]

    m = (right_avg[1] - left_avg[1]) / (right_avg[0] - left_avg[0])
    b = left_avg[1] - m * left_avg[0]
    linear_background = arr[0] * m + b

    return np.asarray([arr[0], arr[1] - linear_background])


def normalize(arr):
    """Normalize the x-y spectra using trapezoidal integration."""
    arr[1] = arr[1] / integrate.trapz(arr[1], arr[0])
    return arr


def plottrim(arr, left, right, relative_position=0):
    """
    Remove left and right edges of specta.

    Trim the x-y spectra to only have x-values between 'left' and 'right'
    Example:
        > arr = [
            [0, 1, 2, 3, 4, 5], # x_values
            [10, 11, 12, 13, 14, 15] # y_values
            ]
        > plottrim(arr, left=2, right=5)
        [[2, 3, 4, 5], [12, 13, 14, 15]]]
    """
    temp = [[], []]
    for i in range(len(arr[0])):
        if np.round(relative_position + left, 6) <= \
           np.round(arr[0][i], 6) <= np.round(relative_position + right, 6):
            temp[0].append(np.round(arr[0][i], 6))
            temp[1].append(np.round(arr[1][i], 6))
    return np.asarray(temp)
