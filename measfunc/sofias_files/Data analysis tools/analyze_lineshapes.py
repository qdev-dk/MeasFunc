from math import floor
import numpy as np 
import scipy.signal 
from general_helpers import find_closest_index

def estimate_full_width_half_maximum(xdata, ydata):
    i_peak = find_closest_index(ydata, ydata.max())
    x_peak = xdata[i_peak]
    i_left = find_closest_index(ydata[0:i_peak], 0.5*(ydata[0:i_peak].max() - ydata[0:i_peak].min()))
    i_right = i_peak + find_closest_index(ydata[i_peak:], 0.5*(ydata[i_peak:].max() - ydata[i_peak:].min()))
    return [i_left, i_peak, i_right]
    
def smooth_with_convolution(data, smooth_percentage):
    w = int(smooth_percentage*len(data))
    kernel = np.ones(w)/w
    start_padd = np.zeros(2*w) + data[0]
    end_padd = np.zeros(2*w) + data[-1]
    #smooth_data = np.convolve(np.concatenate([start_padd, data, end_padd]), kernel, mode='symm')[2*w:-2*w]
    smooth_data = np.convolve(data, kernel, mode='symm') 
    return smooth_data 
    
def find_max_slope_with_smoothing(xdata, ydata, smooth_percentage:float=0.1, decimate:bool=True, peak_height:float=0.1):
    full_result = {}
    w = int(floor(smooth_percentage*len(ydata)))
    #kernel = np.ones(w)/w
    #start_padd = np.zeros(w//2) + ydata[0]
    #end_padd = np.zeros(w - w//2) + ydata[-1]
    #smooth_ydata = np.convolve(np.concatenate([start_padd, ydata, end_padd]), kernel, mode='same')[w//2:-w//2]
    smooth_ydata = smooth_with_convolution(ydata, smooth_percentage)
    dy_dx = np.gradient((smooth_ydata[::w]) if (decimate == True) else smooth_ydata)
    peaks = scipy.signal.find_peaks(dy_dx, height=peak_height*(dy_dx.max() - dy_dx.min()))
    peak_indices = peaks[0]
    peak_heights = peaks[1]['peak_heights']
    
    if (len(peak_indices) == 0):
        print("Warning: peak detection failed")
        best_xvalue = None
    else:
        best_peak_index = peak_indices[list(peak_heights).index(peak_heights.max())]
        best_xvalue = xdata[best_peak_index]

    full_result['best_xvalue'] = best_xvalue 
    full_result['w'] = w
    full_result['best_peak_index'] = best_peak_index 
    full_result['peaks'] = peaks
    full_result['smooth_ydata'] = smooth_ydata
    full_result['dy_dx'] = dy_dx
    return full_result 