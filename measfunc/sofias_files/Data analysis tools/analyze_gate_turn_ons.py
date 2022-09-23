import numpy as np 
import scipy 
from read_plot_qcodes_data import auto_read_to_xy
from general_helpers import find_closest_index 

def find_threshold(xdata, ydata, threshold:float, baseline:float, sweep:str='up'):
    """
    find preset threshold (current)
    """
    if (sweep == 'up'):
        for i in range(len(ydata)):
            if (ydata[i] > (baseline + threshold)):
                return (xdata[i], ydata[i])
    elif (sweep == 'down'):
        for i in range(len(ydata)):
            if (np.flip(ydata)[i] > (baseline + threshold)):
                return (np.flip(xdata)[i], np.flip(ydata)[i])
    return None
# # 
# # 
def find_threshold_and_saturation(xdata, ydata, sigma_scale:float=1.0):
    """
    we assume a classical transistor like turn-on 
    based on evaluating derivative and standard deviation at base level
    no preset thresholds 
    to do: 
    - parameter for i_skip
    - ERROR BARS 
    """    
    # Find thresholds 
    dy_dx = np.gradient(ydata)
    peaks = scipy.signal.find_peaks(dy_dx, width=1)
    peak_indices = peaks[0]
    peak_prominences = peaks[1]['prominences']
    
    for j, ip in enumerate(peak_indices):
        if peak_prominences[j] == max(peak_prominences):
            peak_xy = [xdata[ip], ydata[ip]]
            x_base = xdata[:ip-10]
            dy_dx_base = dy_dx[:ip-10]
            dy_dx_sigma = np.std(dy_dx_base)
            # Find saturations
            # saturation = derivative reaches base level after peak 
            dy_dx_saturation_indices = ip + np.where(
                np.convolve(dy_dx[ip:], np.ones(8), mode='symm') < dy_dx_sigma*sigma_scale)
            i_saturation = dy_dx_saturation_indices[0][0]
            saturation_xy = [xdata[i_saturation], ydata[i_saturation]]
    return peak_xy, saturation_xy
# # 
# # 
def find_threshold_and_saturation_from_run_id(run_id, source_gate_name, sigma_scale:float=1.0):
    """
    we assume a classical transistor like turn-on 
    based on evaluating derivative and standard deviation at base level
    no preset thresholds 
    to do: 
    - parameter for i_skip
    - ERROR BARS 
    """
    dataset = auto_read_to_xy(int(run_id))
    xdata = dataset[dataset['xaxis_name']]
    yaxis_name = [yn for yn in dataset['yaxis_names'] if (source_gate_name in yn)][0]
    ydata = dataset[yaxis_name]
    
    peak_xy, saturation_xy = find_threshold_and_saturation(xdata, ydata, sigma_scale=sigma_scale)
    
    # Find thresholds 
    #dy_dx = np.gradient(ydata)
    #peaks = scipy.signal.find_peaks(dy_dx, width=1)
    #peak_indices = peaks[0]
    #peak_prominences = peaks[1]['prominences']
    
    #for j, ip in enumerate(peak_indices):
    #    if peak_prominences[j] == max(peak_prominences):
    #        peak_xy = [xdata[ip], ydata[ip]]
    #        x_base = xdata[:ip-10]
    #        dy_dx_base = dy_dx[:ip-10]
    #        dy_dx_sigma = np.std(dy_dx_base)
    #        # Find saturations
    #        # saturation = derivative reaches base level after peak 
    #        dy_dx_saturation_indices = ip + np.where(
    #            np.convolve(dy_dx[ip:], np.ones(8), mode='symm') < dy_dx_sigma*sigma_scale)
    #        i_saturation = dy_dx_saturation_indices[0][0]
    #        saturation_xy = [xdata[i_saturation], ydata[i_saturation]]
    #return peak_xy, saturation_xy

def estimate_lever_arm():
    pass 