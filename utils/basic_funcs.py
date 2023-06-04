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

    # Sort based on run number
    alldata_files = sorted(alldata_files, 
                           key=lambda x: int(
                           x.split("_")[-1].replace(".txt","")
                           )
                        )

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

def Voigt(x,A,cen,sigma,gamma):
    # cen: center of the voigt distribution
    # sigma: the standard deviation of the Gaussian
    # gamma: the Half-width at half-maximum of the Lorentzian
    return A*scipy.special.voigt_profile(x-cen,sigma,gamma)


def peak_indices(arr, min_peak_seperation):
    stepsize = np.mean(np.diff(arr[0]))
    peak_index, properties = find_peaks(arr[1], height=0, distance=min_peak_seperation/stepsize)
    peak1 = peak_index[np.argsort(properties["peak_heights"])[-1]] #Ka1 Peak
    peak2 = peak_index[np.argsort(properties["peak_heights"])[-2]] #Ka2 Peak
    return (peak1,peak2)


def fit_Ka_with_voigts(arr, p0=None, num_voigts=2):
    """
    This function fits an arbitrary Kalpha spectrum with 
    any number of voigts. The logic that's used it centered around
    taking best guesses for traditional Kalpha spectra (two main peaks).
    However, by specifying your own initial guesses and number of voits, it
    can fit any arbitrary spectrum.

    The p0 initial guess array is designed to allow for 
    a wide variety of initial guess formats, hopefully reducing
    the work on the user's end when trying to quickly extract a
    fit with Voigts.

    Paramters
    ---------
        arr : 2D numpy array 
            has he form [energy, spectrum]
        p0 : list, optional
            [[voigt1_params], [voigt2_params], ...]

            [voigt_params] can have the format 
            [a, b, c, ...], 
            [[a, amin, amax], [b, bmin, bmax], [c, cmin, cmax], ...],
            or [[a, amin, amax], b, [c, cmin, cmax], ...]
        num_voigts : int, optional
            Number of voigts to use for the fit

    Returns
    -------
        result : lmfit result
            The result from calling model.fit()
        voigts : list
            [voigt1, voigt2, ...]
        
    """

    if num_voigts <= 1:
        raise ValueError("num_voigts must be equal to 2 or greater")


    params = Parameters()
    
    minamp = 0 * np.max(arr[1])
    maxamp = 1000 * np.max(arr[1])
    peakpos = peak_indices(arr, min_peak_seperation = 4)

    if p0: # Initial guesses for parameters are provided

        if len(p0) < num_voigts:
            print("Warning, length of initial guess list 'p0' is less than 'num_voigts'")
            print("Default guess params may not be optiomal for the given spectrum")

        if len(p0) > num_voigts:
            msg = "length of initial guess list 'p0' is greater than 'num_voigts'." 
            raise ValueError(msg)

        v = 0
        for voigt_params in p0:
            for j, p in enumerate(voigt_params):

                if isinstance(p, int) or isinstance(p, float): # p has format [a]

                    shift = 2
                    if j == 0:
                        params.add("amp" + str(v),value=p,min=minamp,max=maxamp)
                    elif j == 1:
                        params.add("cen" + str(v),value=p,min=arr[0][peakpos[0]]-shift,max=arr[0][peakpos[0]]+shift)
                    elif j == 2:
                        params.add("sigma" + str(v),value=p,min=0,max=1000)
                    elif j == 3:
                        params.add("gamma" + str(v),value=p,min=0,max=1000)

                elif isinstance(p, list) and len(p) == 3: # p has format [a, amin, amax]

                    if j == 0:
                        params.add("amp" + str(v),value=p[0],min=p[1],max=p[2])
                    elif j == 1:
                        params.add("cen" + str(v),value=p[0],min=p[1],max=p[2])
                    elif j == 2:
                        params.add("sigma" + str(v),value=p[0],min=p[1],max=p[2])
                    elif j == 3:
                        params.add("gamma" + str(v),value=p[0],min=p[1],max=p[2])
                
                elif not(p): # p is None

                    shift = np.abs(arr[0][peakpos[1]] - arr[0][peakpos[0]]) * 1.5
                    if j == 0:
                        params.add("amp" + str(v),value=np.max(arr[1])/2,min=minamp,max=maxamp)
                    elif j == 1:
                        middle = (arr[0][peakpos[1]] - arr[0][peakpos[0]])/2
                        params.add("cen" + str(v),value=middle,min=middle-shift,max=middle+shift)
                    elif j == 2:
                        params.add("sigma" + str(v),value=1,min=0,max=1000)
                    elif j == 3:
                        params.add("gamma" + str(v),value=1,min=0,max=1000)

                else:
                    msg = "p0 must have the format [[voigt1_params], [voigt2_params], ...]"
                    msg += "\nwhere voigt_params must have the format [a, b, c, ...]"
                    msg += "\nor [[a, amin, amax], [b, bmin, bmax], [c, cmin, cmax], ...]"
                    msg += "\nor [[a, amin, amax], b, [c, cmin, cmax], ..."
                    msg += "\nwhere a, b, and c are initial guesses."
                    raise ValueError(msg)
            v += 1


    else: # No initial guesses for parameters

        if num_voigts > 2:
            print("Warning, fitting more than 2 voigts with no initial guesses",
                "\nThis will be difficult to converge without " + 
                "strong initial conditions")

        shift = 3
        v = 2

        params.add("amp0",value=arr[1][peakpos[0]],min=minamp,max=maxamp)
        params.add("amp1",value=arr[1][peakpos[1]],min=minamp,max=maxamp)
        params.add("cen0",value=arr[0][peakpos[0]],min=arr[0][peakpos[0]]-shift,max=arr[0][peakpos[0]]+shift)
        params.add("cen1",value=arr[0][peakpos[1]],min=arr[0][peakpos[1]]-shift,max=arr[0][peakpos[1]]+shift)
        params.add("sigma0",value=1,min=0,max=1000)
        params.add("sigma1",value=1,min=0,max=1000)
        params.add("gamma0",value=1,min=0,max=1000)
        params.add("gamma1",value=1,min=0,max=1000)


    if v < num_voigts:
        
        new_voigts = num_voigts - v

        centers = [params[x].value for x in params.keys() if "cen" in x]
        centers = np.sort(np.asarray(centers))
        largest_gap = max(centers[i+1] - centers[i] for i in range(len(centers) - 1))
        new_centers = [centers[0] + largest_gap / (new_voigts + 1)]

        for cen in new_centers:

            shift = largest_gap
            params.add("amp" + str(v),value=np.max(arr[1])/2,min=minamp,max=maxamp)
            params.add("cen" + str(v),value=cen,min=cen-shift,max=cen+shift)
            params.add("sigma" + str(v),value=1,min=0,max=1000)
            params.add("gamma" + str(v),value=1,min=0,max=1000)
            
            v += 1

    def voigtmodel(x, **kwargs):
        spectra = np.zeros(len(arr[0]))
        for i in range(num_voigts):
            spectra += Voigt(x,A=kwargs["amp" + str(i)], cen=kwargs["cen" + str(i)],
                             sigma=kwargs["sigma" + str(i)], gamma=kwargs["gamma" + str(i)])
        return spectra

    result = Model(voigtmodel).fit(data=arr[1],x=arr[0],params=params)

    voigts = []
    for v in range(num_voigts):
        voigt = Voigt(arr[0], A=result.params['amp' + str(v)].value, cen=result.params['cen' + str(v)].value, 
                      sigma=result.params['sigma' + str(v)].value, gamma=result.params['gamma' + str(v)].value)
        voigts.append(voigt)

    return result, voigts