import os
import numpy as np
import pandas as pd

from scipy import integrate
from scipy import optimize


def get_spectra(run_path,runs):
    ''' 
    Read in an 'alldata' file, sum together runs (either a 
    list of runs or 'all'), and return a numpy array of the form
    [[x_values], [y_values]].
    '''

    alldata_files = [f.path for f in os.scandir(run_path) if "alldata" in str(f.path)]

    for key,line in enumerate(open(alldata_files[0]).readlines()):
        if line.startswith("***"):
            rows_to_skip = key + 1

    data = [pd.read_csv(x,delim_whitespace=True,skiprows=rows_to_skip) for x in alldata_files]
    x = data[0]["Energy_(eV)"]

    if isinstance(runs,str) and runs == "all":
        ''' Return a sum of all runs '''

        y = np.asarray([d["cnts_per_live"] for d in data])
        return np.asarray([x,np.sum(y,axis=0)])

    elif isinstance(runs,list):
        ''' Returns a sum of the runs specified in the list 'runs' '''

        y = np.asarray([dat["cnts_per_live"] for key,dat in enumerate(data) if key in runs])
        return np.asarray([x,np.sum(y,axis=0)])

    else:
        raise ValueError("runs must be either string 'all' or list (ex:[0,1,2]) denoting the which specific runs you wish to sum together")

def find_index_of_closest_value(array, val):
    dif = (array - val)**2
    return np.argmin(dif)

def subtract_constant_background(arr, roi):
    ''' 
    Subtract a constant background from the x-y spectra using 
    the intensity within the region of interest (ROI).
    '''

    start_energy, end_energy = roi

    def loss(y, b):
        fit_function = b # constant background
        return np.sum((y - fit_function)**2 / 100) 
  
    start_index = find_index_of_closest_value(arr[0], start_energy)
    end_index = find_index_of_closest_value(arr[0], end_energy)
    y = arr[1][start_index:end_index] 

    starting_param_vals = [0]  # starting guess of zero background

    optimized_background = optimize.minimize(loss, x0=starting_param_vals, 
        args=(y), method='BFGS')
    background = optimized_background['x']

    return np.asarray([arr[0],arr[1]-background])

def subtract_linear_background(arr, left_roi, right_roi):
    '''
    Subtract linear background from the x-y spectra. Find 
    average x and y values within the two ROIs and use the 
    line connecting them.
    '''

    left_start_energy, left_end_energy = left_roi
    right_start_energy, right_end_energy = right_roi
    
    left_chunk = plottrim(arr, left_start_energy, left_end_energy)
    right_chunk = plottrim(arr, right_start_energy, right_end_energy)

    left_avg = np.average(left_chunk,axis=1) # [x_{1,avg}, y_{1,avg}]
    right_avg = np.average(right_chunk,axis=1) # [x_{2,avg}, y_{2,avg}]

    m = (right_avg[1] - left_avg[1]) / (right_avg[0] - left_avg[0])
    b = left_avg[1] - m*left_avg[0]
    linear_background = arr[0]*m + b

    return np.asarray([arr[0],arr[1]-linear_background])

def normalize(arr):
    ''' Normalize the x-y spectra using trapezoidal integration '''

    arr[1] = arr[1]/integrate.trapz(arr[1],arr[0])
    return arr
    
def plottrim(arr, left, right, relative_position=0):
    ''' 
    Trim the x-y spectra to only have x-values between 'left' and 'right' 
    Example:
        > arr = [
            [0, 1, 2, 3, 4, 5], # x_values
            [10, 11, 12, 13, 14, 15] # y_values
            ]
        > plottrim(arr, left=2, right=5)
        [[2, 3, 4, 5], [12, 13, 14, 15]]]
    '''

    temp = [[],[]]
    for i in range(len(arr[0])):
        if(np.round(relative_position + left,6) <= np.round(arr[0][i],6) <= np.round(relative_position + right,6)):
            temp[0].append(np.round(arr[0][i],6))
            temp[1].append(np.round(arr[1][i],6))
    return np.asarray(temp)

def flip(arr):
    ''' Flip an x-y spectra across the y axis '''

    temp = [[],[]]
    for i in range(len(arr[0])):
        temp[0].append(arr[0][i]*-1)
        temp[1].append(arr[1][i])
    arr = np.asarray(temp)
    arr = arr.T
    arr = arr[np.argsort(arr[:,0])]
    arr = arr.T
    return arr