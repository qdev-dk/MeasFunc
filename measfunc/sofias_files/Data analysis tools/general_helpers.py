import numpy as np 
from math import floor 
import scipy.signal 
def find_closest_index(data, value):
    distances = np.abs(np.array(data) - value)
    return list(distances).index(distances.min())

def find_matrix_extremum_indices(z:np.ndarray, extremum:str='max'):
    if (extremum not in ['min', 'max']):
        raise ValueError("Find either min or max")

    z_extremum = z[0,0]
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if (extremum == 'max'):
                if (z[i,j] == z.max()):
                    i_extremum = i
                    j_extremum = j
                    z_extremum = z[i_extremum,j_extremum]
                    break
            if (find == 'min'):
                if (z[i,j] == z.min()):
                    i_extremum = i
                    j_extremum = j
                    z_extremum = z[i_extremum,j_extremum]
                    break
    return [i_extremum, j_extremum, z_extremum] 

def get_data_along_line(xdata:np.ndarray, ydata:np.ndarray, zdata:np.ndarray, slope:float, y0:float):
    """
    """
    if (zdata.shape[0] != len(ydata)) or (zdata.shape[1] != len(xdata)):
        raise ValueError("Check array shapes")

    xy = []
    ztrace = np.zeros(len(xdata))
    for i, x_i in enumerate(xdata):
        y_along_line = slope*x_i + y0 
        closest_y_index = find_closest_index(ydata, y_along_line)
        closest_y = ydata[closest_y_index]
        if (closest_y < ydata.min()) or (closest_y > ydata.max()):
            xy.append([None, None])
            continue 
        ztrace[i] = zdata[closest_y_index, i]
        xy.append([x_i, closest_y])
    return xy, ztrace 
    
def restack_zdata(zdata, index:int, axis:int=1):
    """
    """
    if ((axis != 0) and (axis != 1)):
        raise ValueError("Axis needs to be 0 or 1") 
    if (axis == 1):
        zdata_1 = zdata[:, 0:index]
        zdata_2 = zdata[:, index:]
        return np.concatenate([zdata_2, zdata_1], axis=1)
    elif (axis == 0):
        zdata_1 = zdata[0:index, :]
        zdata_2 = zdata[index:, :]
        return np.concatenate([zdata_2, zdata_1], axis=0)

def smooth(y, window_size, order=3, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Taken from: http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in
                range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')
        