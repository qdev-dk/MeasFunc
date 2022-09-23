import numpy as np

def sweep_integration_time(data:np.ndarray):
    """
    """
    integrated_data = np.zeros(len(data)-1)
    for i_t in range(1,len(data)):
        integrated_data[i_t-1] = np.average(data[0:i_t])
    return integrated_data 
    