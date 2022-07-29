import os
import numpy as np
import pandas as pd

from scipy import integrate
from scipy import optimize

def get_spectra(run_path,runs,rows_to_skip):
    ''' 
    Read in output from the rattrap spectrometer in Seidler lab,
    and sum multiple runs together if there is more than 1.
    '''

    data_files = [f.path for f in os.scandir(run_path) if "alldata" in str(f.path)]
    data = [pd.read_csv(x,delim_whitespace=True,skiprows=rows_to_skip) for x in data_files]
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

def background_subtract(arr,start_energy):
    ''' 
    Subtract the background from the x-y spectra using 
    the intensity of the at the farthest tip of the tail.
    '''

    def find_index_of_closest_value(array, val):
        dif = (array - val)**2
        return np.argmin(dif)

    def loss(y, b):
        return np.sum((y - b)**2/100)

    # background subtract from tail
    start_index = find_index_of_closest_value(arr[0], start_energy)
    # take the last values of the spectrum starting at the specified energy
    y = arr[1][start_index:-1] 

    starting_param_vals = np.array([0.0])  #constant background

    optimized_background = optimize.minimize(loss, x0=starting_param_vals, 
        args=(y), method='BFGS')
    print(optimized_background['message'])
    background = optimized_background['x']

    return np.asarray([arr[0],arr[1]-background])

def normalize(arr):
    ''' Normalize the x-y spectra '''

    arr[1] = arr[1]/integrate.trapz(arr[1],arr[0])
    return arr
    
def plottrim(arr,left,right,relative_position=0):
    ''' Trim the x-y spectra to only have x-values between 'left' and 'right' '''

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