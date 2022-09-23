import numpy as np
import scipy.signal 

def get_subthreshold_zdata(zdata:np.ndarray, z_threshold=0.1):
    threshold_is_float = type(z_threshold) == type(0.1)
    threshold_is_list = type(z_threshold) == type([1.0,2.0])
    if (not threshold_is_float) and (not threshold_is_list):
        raise ValueError("Invalid z_threshold type. Must be float or list")
    if (threshold_is_float):
        zth = np.abs(z_threshold)
        subthreshold_zdata = np.ma.masked_array(zdata, mask=((zdata > zth) + (zdata < -zth)))
    elif threshold_is_list:
        if (len(z_threshold) != 2):
            raise ValueError("Invalid z_threshold length. Please provide [zmin, zmax]")
        zmin = min(z_threshold)
        zmax = max(z_threshold)
        subthreshold_zdata = np.ma.masked_array(zdata, mask=((zdata > zmax) + (zdata < zmin)))
    return subthreshold_zdata

def threshold_edges(xdata:np.ndarray, ydata:np.ndarray, zdata:np.ndarray, 
    z_threshold=0.1):
    """
    Define masked array from zdata, retaining values below a threshold (masking those above)
    Find edges of the masked array 
    """
    subthreshold_zdata = get_subthreshold_zdata(zdata, z_threshold=z_threshold)
    ny, nx = zdata.shape
    edge_indices_along_y = []
    edge_xy = []
    for ix in range(nx):
        xedge_indices = np.ma.notmasked_edges(subthreshold_zdata[:,ix])
        edge_indices_along_y.append(xedge_indices)
        for iy in xedge_indices:
            edge_xy.append([xdata[ix], ydata[iy]])
    return edge_xy

def autofind_diamond_peaks(xdata:np.ndarray, ydata:np.ndarray, zdata:np.ndarray, 
    z_threshold=0.1, y_threshold:float=0.0, smooth:bool=True, peak_properties:dict={}):
    """
    z_threshold: threshold current (no current, on-current)
        - we assumem no offset currents 
    y_threshold: "symmetry point", offset bias voltage 
    """
    edge_xy = threshold_edges(xdata, ydata, zdata, z_threshold=z_threshold)
        
    num_clusters = 2
    peak_clusters = [[] for i in range(num_clusters)]
    for i in range(num_clusters): 
        points = []
        # Group edges into groups (positive, negative source-drain voltage)
        for xy in edge_xy:
            if ((i == 0) and (xy[1] > y_threshold)):
                points.append(xy)
            elif (i == 1) and (xy[1] <= y_threshold):
                points.append(xy)
        x, y = zip(*points)
        x = np.array(x)
        y = np.array(y)
        if (smooth == True):
            w = 4
            y = (np.convolve(y, np.ones(w), mode='symm')/w)[w//2:-w//2]
            x = x[w//2:-w//2]
        # Find peaks 
        if (i == 0):
            peak_indices, _ = scipy.signal.find_peaks(y, **peak_properties) 
        else:
            peak_indices, _ = scipy.signal.find_peaks(np.abs(y), **peak_properties)  
        peak_xy = []
        for ip in peak_indices:
            peak_xy.append([x[ip], y[ip]])
        peak_clusters[i] = peak_xy
    return peak_clusters 

def autoestimate_charging_energies_lever_arms(xdata:np.ndarray, ydata:np.ndarray, 
    zdata:np.ndarray, z_threshold=0.1, y_threshold:float=0.0, smooth:bool=True,
    peak_properties:dict={}):
    """
    E_addition = e^2/C := 2 E_C 
    E_addition = top to middle, or middle to bottom of a Coulomb diamond = h_tm = h_mb
    
    dot-dotgate lever arm: 
    e^2/(alpha C) = (half heifht) / (full width of a CD) = (h/2)/w
    <-> alpha = w*C/e^2 = w/E_addition 
    """
    peak_clusters = autofind_diamond_peaks(xdata, ydata, zdata, 
        z_threshold=z_threshold, y_threshold=y_threshold, smooth=smooth,
        peak_properties=peak_properties)

    half_heights = []
    widths = []
    lever_arms = [[] for ic in range(2)]
    charging_energies = [[] for ic in range(2)]
    for ic in range(2):
        for i, xy in enumerate(peak_clusters[ic]):
            half_heights.append(xy[1] - y_threshold)
            if (i > 0):
                widths.append(xy[0] - peak_clusters[ic][i-1][0])
                lever_arms[ic].append(half_heights[-1]/widths[-1])
                charging_energies[ic].append(half_heights[-1])
    return charging_energies, lever_arms 