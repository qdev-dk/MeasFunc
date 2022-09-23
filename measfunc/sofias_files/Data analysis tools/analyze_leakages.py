import numpy as np 
from general_helpers import find_closest_index 

def find_xaxis_matching_run_id(datasets, run_ids, xaxis_name):
    for run_id in run_ids:
        if (xaxis_name in datasets[run_id]['xaxis_name']):
            #print(data_label)
            return run_id
    return None

def get_leakage_matrix(datasets, run_ids, Aunits=1.0, Vunits=1.0, vrange:list=[]):
    """
    rows: current
    columns: voltage 
    """
    i = 1
    channels = {}
    for run_id in run_ids:
        xaxis_name = datasets[run_id]['xaxis_name']
        instrument, gate_name = xaxis_name.split("_V_")
        channels[gate_name] = {"index": i, "instrument": instrument}
        i += 1

    leakage_matrix = np.zeros((len(channels.keys()), len(channels.keys())))

    for xgate_name in channels.keys():
        xchannel_index = channels[xgate_name]["index"]
        xinstrument = channels[xgate_name]["instrument"]
        
        xaxis_name = [k for k, v in channels.items() 
            if (v["index"] == xchannel_index)][0]
        xaxis_name = xinstrument+"_V_"+xaxis_name
        run_id = find_xaxis_matching_run_id(datasets, run_ids, xaxis_name)
        xdata = datasets[run_id][xaxis_name]
        ix1 = find_closest_index(xdata, vrange[0])
        ix2 = find_closest_index(xdata, vrange[1])
        if (len(vrange) == 2):
            xdata = xdata[ix1:ix2]

        for ygate_name in channels.keys():    
            ychannel_index = channels[ygate_name]["index"]
            yinstrument = channels[ygate_name]["instrument"]
            ychannel_name = [k for k, v in channels.items() 
                if (v["index"] == ychannel_index)][0]
            yaxis_name = yinstrument+"_I_"+ychannel_name
            ydata = datasets[run_id][yaxis_name]
            if (len(vrange) == 2):
                ydata = ydata[ix1:ix2]
            
            leakage_matrix[ychannel_index-1,xchannel_index-1] = np.average(np.gradient(ydata))
            
    leakage_matrix = leakage_matrix*Vunits/Aunits
    return channels, leakage_matrix 