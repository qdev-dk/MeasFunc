from math import floor 
import numpy as np 
import scipy 
from scipy import signal

from general_helpers import find_closest_index, smooth 

def estimate_coulomb_diamonds(xdata, ydata, zdata, **kwargs):
    """
    kwargs:
        curvature_smoothing_percentage: float between 0 and 1. 
                                        Adjust d2z_dy2 smoothness 
        upper_num_sigma: positive float. 
                         Adjust how prominent PEAKS in d2z_dy2 are saved to TOP EDGES of Coulomb diamonds.
        lower_num_sigma: positive float. 
                         Adjust how prominent DIPS in d2z_dy2 are saved to BOTTOM EDGES of Coulomb diamonds.      
    """
    # # 
    # # kwargs 
    if ('curvature_smoothing_percentage' in kwargs.keys()):
        curvature_smoothing_percentage = kwargs['curvature_smoothing_percentage']
    else: 
        curvature_smoothing_percentage = 0.05 
        
    if ('upper_num_sigma' in kwargs.keys()):
        upper_num_sigma = kwargs['upper_num_sigma']
    else: 
        upper_num_sigma = 1 
        
    if ('lower_num_sigma' in kwargs.keys()):
        lower_num_sigma = kwargs['lower_num_sigma']
    else: 
        lower_num_sigma = 1
    
    if ('smooth_top_edges' in kwargs.keys()):
        smooth_top_edges = kwargs['smooth_top_edges']
    else:
        smooth_top_edges = True 
        
    if ('smooth_bottom_edges' in kwargs.keys()):
        smooth_bottom_edges = kwargs['smooth_bottom_edges']
    else:
        smooth_bottom_edges = True 
        
    if ('diamond_peak_width' in kwargs.keys()):
        diamond_peak_width = kwargs['diamond_peak_width']
    else: 
        diamond_peak_width = 5
        
    if ('diamond_dip_width' in kwargs.keys()):
        diamond_dip_width = kwargs['diamond_dip_width']
    else: 
        diamond_dip_width = 5

    d2z_dy2_matrix = [[] for i in enumerate(xdata)]
    top_edges = [] 
    bottom_edges = []
    for i, x_i in enumerate(xdata):
        # # 
        # # Calculate second derivative, with smoothing 
        #   (noise amplifies a lot when taking derivatives)
        w = int(floor(curvature_smoothing_percentage*len(ydata)))
        if (w % 2 == 0):
            w += 1 
        dz_dy = np.diff(smooth(zdata[:, i], window_size=w), n=1)
        d2z_dy2 = np.diff(smooth(dz_dy, window_size=w), n=1)
        #istart = 8
        #istop = -8
        #dz_dy = (np.convolve(np.diff(zdata[:, i], n=1), np.ones(w), mode='symm')/w)[2*istart:istop]
        #d2z_dy2 = (np.convolve(np.diff(dz_dy, n=1), np.ones(w), mode='symm')/w)[2*istart:istop]
        d2z_dy2_matrix[i] = d2z_dy2 # for debugging 

        i_min = find_closest_index(d2z_dy2, d2z_dy2.min())
        i_max = find_closest_index(d2z_dy2, d2z_dy2.max())
        y_at_min_curvature = ydata[i_min] #ydata[4*istart:2*istop][i_min]
        y_at_max_curvature = ydata[i_max] #ydata[4*istart:2*istop][i_max]

        d2z_dy2_average = np.average(d2z_dy2[i_min:i_max]) # estimate average at zero current regime 
        d2z_dy2_sigma = np.std(d2z_dy2[i_min:i_max]) # estimate sigma in zero current regime 

        if (d2z_dy2.max() > (d2z_dy2_average + upper_num_sigma*d2z_dy2_sigma)):
            top_edges.append([x_i, y_at_max_curvature])
            
        if (d2z_dy2.min() < (d2z_dy2_average - lower_num_sigma*d2z_dy2_sigma)):
            bottom_edges.append([x_i, y_at_min_curvature])
    
    d2z_dy2_matrix = np.array(d2z_dy2_matrix)
    top_edges = np.array(top_edges)
    bottom_edges = np.array(bottom_edges)
    
    # # 
    # # Find diamond tops (peaks), bottoms (dips)
    w = 2 
    if (smooth_top_edges is True):
        diamond_peaks, _ = scipy.signal.find_peaks(
            np.convolve(top_edges[:,1], np.ones(w), mode='symm')/w, width=diamond_peak_width)
    else:
        diamond_peaks, _ = scipy.signal.find_peaks(top_edges[:,1], width=diamond_peak_width)
    diamond_peaks = np.array(list(zip(top_edges[:,0][diamond_peaks], top_edges[:,1][diamond_peaks])))
        
    if (smooth_bottom_edges is True):
        diamond_dips, _ = scipy.signal.find_peaks(
            np.convolve(-1*bottom_edges[:,1], np.ones(w), mode='symm')/w, width=diamond_dip_width)
    else:
        diamond_dips, _ = scipy.signal.find_peaks(-1*bottom_edges[:,1], width=diamond_dip_width)
    diamond_dips = np.array(list(zip(bottom_edges[:,0][diamond_dips], bottom_edges[:,1][diamond_dips])))
        
    return diamond_dips, diamond_peaks, bottom_edges, top_edges
    
def estimate_charging_energies_lever_arms(diamond_dips, diamond_peaks):
    """
    """
    half_heights = []
    widths = []
    lever_arms = []
    charging_energies = []
    
    i_difference = 0 
    for i, (dd_xy, dp_xy) in enumerate(list(zip(diamond_dips, diamond_peaks))):
        #for i, xy in enumerate(peak_clusters[ic]):
        half_heights.append((dp_xy[1] - dd_xy[1 + i_difference])/2) #xy[1] - y_threshold)
        if (i > 0):
            widths.append((dp_xy[0] - diamond_peaks[:,0][i-1])/2 + (dd_xy[0] - diamond_dips[:,0][i-1])/2)
            lever_arms.append(half_heights[-1]/widths[-1])
            charging_energies.append(half_heights[-1])
    return charging_energies, lever_arms 