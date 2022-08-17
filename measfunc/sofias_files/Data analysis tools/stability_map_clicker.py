import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scipy.signal 
import skimage.feature 
#from skimage.feature import canny

def line_parameters(p1:float, p2:float):
    """
    p1, p2 are two points along the line
    """
    slope = (p2[1] - p1[1])/(p2[0] - p1[0])
    y0 = p1[1] - slope*p1[0]
    return [slope, y0]

def line_function(x:float, slope:float, y0:float):
    return slope*x + y0

def inverse_line_function(y:float, slope:float, y0:float):
    return (y - y0)/slope

def get_subthreshold_zdata(zdata:np.ndarray, z_threshold:float=0.1):
    zth = np.abs(z_threshold)
    subthreshold_zdata = np.ma.masked_array(zdata, mask=((zdata > zth) + (zdata < -zth)))
    return subthreshold_zdata

def threshold_edges(xdata:np.ndarray, ydata:np.ndarray, zdata:np.ndarray, 
    z_threshold:float=0.1):
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
    z_threshold:float=0.1, y_threshold:float=0.0, smooth:bool=False):
    """
    z_threshold: threshold current (no current, on-current)
        - we assumem no offset currents 
    y_threshold: "symmetry point", offset bias voltage 
    """
    edge_xy = threshold_edges(xdata, ydata, zdata, z_threshold=0.1)
        
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
            peak_indices, _ = scipy.signal.find_peaks(y) # find maxima of positive-valued data 
        else:
            peak_indices, _ = scipy.signal.find_peaks(np.abs(y)) # find minima of negative-valued data 
        peak_xy = []
        for ip in peak_indices:
            peak_xy.append([x[ip], y[ip]])
        peak_clusters[i] = peak_xy
    return peak_clusters 

def autoestimate_charging_energies_lever_arms(xdata:np.ndarray, ydata:np.ndarray, 
    zdata:np.ndarray, z_threshold:float=0.1, y_threshold:float=0.0):
    """
    E_addition = e^2/C := 2 E_C 
    E_addition = top to middle, or middle to bottom of a Coulomb diamond = h_tm = h_mb
    
    dot-dotgate lever arm: 
    e^2/(alpha C) = full width of a CD = w
    <-> alpha = w*C/e^2 = w/E_addition 
    """
    peak_clusters = autofind_diamond_peaks(xdata, ydata, zdata, 
        z_threshold=z_threshold, y_threshold=y_threshold)

    charging_energies = [[] for ipc in range(2)]
    for ic in range(2):
        for xy in peak_clusters[ic]:
            charging_energies[ic].append(np.abs(xy[1] - y_threshold))

    lever_arms = [[] for ic in range(2)]
    for ic in range(2):
        for ip in range(len(peak_clusters[ic])-1):
            xy_N = peak_clusters[ic][ip]
            xy_Np1 = peak_clusters[ic][ip+1]
            E_C = charging_energies[ic][ip]
            lever_arms[ic].append((xy_Np1[0] - xy_N[0])/(2*E_C))
    return charging_energies, lever_arms 

def edge_positions(xdata:np.ndarray, ydata:np.ndarray, zdata:np.ndarray, sigma:float=1.0):
    """
    """
    ny, nx = zdata.shape
    edges = skimage.feature.canny(zdata, sigma=sigma)
    
    edge_xy = [] 
    for i in range(nx):
        for j in range(ny):
            if (edges[j,i] == True):
                edge_xy.append([xdata[i], ydata[j]])
    edge_xy = np.array(edge_xy)
    return edge_xy

def get_edge_points(xdata:np.ndarray, ydata:np.ndarray, zdata:np.ndarray, 
    zdata_format:str="dz_dx", sigma:float=1.0):
    dz_dy, dz_dx = np.gradient(zdata)
    if (zdata_format == "dz_dx"):
        edge_points = edge_positions(xdata, ydata, dz_dx, sigma=sigma)
    elif (zdata_format == "dz_dy"):
        edge_points = edge_positions(xdata, ydata, dz_dy, sigma=sigma)
    else:
        edge_points = edge_positions(xdata, ydata, zdata, sigma=sigma)
    return edge_points

def find_closeby_points(line_points, data_points, distance_threshold:float=0.01, 
    epsilon_threshold:float=0.01):
    """
    find points from data_points that fall close to a line defined by line_points
    """
    # Helper variables
    # Full dataset ranges -> use for distance threshold
    xrange = data_points[:,0] 
    yrange = data_points[:,1] 
    slope, y0 = line_parameters(line_points[0], line_points[1])
    
    # Boxes based on clicked points
    xbox = np.array([line_points[0][0], line_points[1][0]])
    xbox = np.array([xbox.min(), xbox.max()])
    dx = xbox.max() - xbox.min()
    
    ybox = np.array([line_points[0][1], line_points[1][1]])
    ybox = np.array([ybox.min(), ybox.max()])
    dy = ybox.max() - ybox.min()
    
    box_ratio = dy/dx
    
    # Small extra allowed ranges (helps with e.g. vertical lines)
    epsilon_x = np.abs(xrange.max() - xrange.min())*epsilon_threshold
    epsilon_y = np.abs(yrange.max() - yrange.min())*epsilon_threshold
    xbox = np.array([xbox.min() - epsilon_x, xbox.max() + epsilon_x])
    ybox = np.array([ybox.min() - epsilon_y, ybox.max() + epsilon_y])
    
    num_within_box = 0
    closeby_points = []
    for xy in data_points:
        x = xy[0]
        y = xy[1]
        if ((x < xbox.min()) or (x > xbox.max()) 
            or (y < ybox.min()) or (y > ybox.max())):
            continue
        #print("stopping to analyze the point xy = ",xy)
        if (box_ratio < 1.0):
            euclidean_distance = np.sqrt((line_function(x, slope, y0) - y)**2)
            if (euclidean_distance <= np.abs(yrange.max() - yrange.min())*distance_threshold):
                closeby_points.append(xy)
        else:
            euclidean_distance = np.sqrt((inverse_line_function(y, slope, y0) - x)**2)
            if (euclidean_distance <= np.abs(xrange.max() - xrange.min())*distance_threshold):
                closeby_points.append(xy)
        num_within_box += 1
    return closeby_points, xbox, ybox, num_within_box

def classify_edges(num_up:int, num_down:int, xdata:np.ndarray, ydata:np.ndarray, 
    zdata:np.ndarray, ax, zdata_format:str="dz_dx", sigma:float=1.0, 
    distance_threshold:float=0.01, verbose:bool=False):
    """
    """
    # Find edge data points (choose how to process zdata)
    edge_points = get_edge_points(xdata, ydata, zdata, 
        zdata_format=zdata_format, sigma=sigma)
        
    for xy in edge_points:
        ax.plot(xy[0], xy[1], '.', color='C1', markersize=2)
      
    # Classify edge points to up and down slopes
    point_clusters = [[] for i in range(2)] # 2 for up and down sweeps
    for j in range(2):
        num_points = [num_up, num_down][j]
        for i in range(num_points):
            help_text = "Find "+str(i+1)+" / "+str(num_points)+" "
            help_text += ("up" if (j == 0) else "down")+" slopes"
            ax.set_title(help_text)
            ax.figure.canvas.draw()
            line_points = plt.ginput(n=2, show_clicks=True, timeout=10)
            ax.plot([line_points[0][0], line_points[1][0]], 
                    [line_points[0][1], line_points[1][1]], 
                    dashes=[6, 2], color='black')
        
            closeby_points, xbox, ybox, num_within_box = find_closeby_points(
                line_points, edge_points, distance_threshold=distance_threshold)
            if (verbose == True):
                print("Found ",len(closeby_points),"closeby points")
                print("xbox: ", xbox)
                print("ybox: ", ybox)
                print("num within box: ",num_within_box)
            rect = matplotlib.patches.Rectangle((xbox.min(), ybox.min()), 
                np.abs(xbox.max() - xbox.min()), np.abs(ybox.max() - ybox.min()), 
                linewidth=0.5, edgecolor='C0', facecolor='none')
            ax.add_patch(rect)
            for xy in closeby_points:
                ax.plot(xy[0], xy[1], '.', color='C3', markersize=2)
            point_clusters[j].append(closeby_points)
            
            if ((j == 1) and (i == num_points-1)):
                help_text = "All done! You can close this now"
                ax.set_title(help_text)
                ax.figure.canvas.draw()
    return ax, point_clusters