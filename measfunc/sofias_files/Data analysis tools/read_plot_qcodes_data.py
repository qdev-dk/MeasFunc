import numpy as np
import pandas as pd
import qcodes as qc
from qcodes import load_by_id
from general_helpers import find_closest_index 

def find_run_ids(exc, experiment_name, sample_name):
    run_ids = []
    for experiment in exc.experiments():
        if ((experiment.name == experiment_name) 
            and (experiment.sample_name == sample_name)):
            for data_set in experiment.data_sets():
                run_ids.append(str(data_set.run_id))
    return run_ids

def get_channel_voltage_from_snapshot(dataset, channel:int, qdac='qdac1'):
    qdac_metadata = dataset['dataset'].snapshot['station']['instruments'][qdac]
    channel_parameters = qdac_metadata['submodules']['ch'+str(channel)]['parameters']
    voltage_value = channel_parameters['v']['value']
    return voltage_value

def find_first_nan_index(data):
    for i, val in enumerate(data):
        if pd.isna(data[i]):
            break 
    return i 

def find_first_value_change_index(data):
    old_value = data[0]
    for i, val in enumerate(data):
        new_value = data[i]
        if (new_value != old_value):
            break 
    return i 

def find_slow_variable_trace_indices(y):
    new_trace_indices = []
    y_old = y[0]
    for i_y, y_i in enumerate(y):
        y_new = y_i 
        if (y_new != y_old):
            new_trace_indices.append(i_y)
        y_old = y_new 
    new_trace_indices += [len(y)-1]
    return new_trace_indices

def read_df_to_xy(df, xname, yname, xrange='auto', yrange='auto'):
    if (type(yname) == list):
        df_flat = []
        for i in range(len(yname)):
            df_flat.append(df[yname[i]].reset_index())
        x = df_flat[0][xname]
    else:
        df_flat = df[yname].reset_index()
        x = df_flat[xname]
    if (type(yname) == list):
        y = []
        for i in range(len(yname)):
            y.append(df_flat[i][yname[i]])
    else:
        y = df_flat[yname]
    if (xrange == 'auto'):
        xrange = [0,len(x)] # use iloc instead of loc
    x = x[xrange[0]:xrange[1]]
    x = np.unique(np.array(x))
    if (type(yname) == list):
        y[i] = np.array(y[i])
    else:
        y = np.array(y)
    return (x,y)

def auto_read_to_xy(run_id:int):
    """
    Determine x and y axes automatically
    """
    dataset = {}
    dataset['dataset'] = load_by_id(run_id)
    try:
        dataset['df'] = dataset['dataset'].to_pandas_dataframe_dict()
    except:
        dataset['df'] = dataset['dataset'].get_data_as_pandas_dataframe()
    
    xaxis_name = get_axis_name(dataset['df'])
    dataset[xaxis_name] = []
    dataset['xaxis_name'] = xaxis_name
    
    yaxis_names = []
    for yaxis_name in list(dataset['df'].keys()):
        dataset[yaxis_name] = []
        yaxis_names.append(yaxis_name)
    dataset['yaxis_names'] = yaxis_names
    
    for i in range(len(yaxis_names)):
        [dataset[xaxis_name], dataset[yaxis_names[i]]] = read_df_to_xy(
            dataset['df'], xname=xaxis_name, yname=yaxis_names[i])
    return dataset

def read_df_to_xz_traces(df, xname, yname, zname):
    #for i_variable, zname in enumerate(zaxis_names):
    if (type(zname) == list):
        df_flat = []
        for z_parameter in zname:
            df_flat.append(df[z_parameter].reset_index())
        x = df_flat[0][xname]
        y = df_flat[0][yname]
    else:
        df_flat = df[zname].reset_index()
        x = df_flat[xname]
        y = df_flat[yname]

    if (type(zname) == list):
        z = []
        for i_parameter, z_parameter in enumerate(zname):
            z.append(df_flat[i_parameter][z_parameter])
    else:
        z = df_flat[zname]

    trace_indices = find_slow_variable_trace_indices(y)
    x_traces = []
    z_traces = []
    for i,j in zip([0] + trace_indices[:-1], trace_indices):
        #print(i,j)
        x_traces.append(np.array(x)[i:j])
        z_traces.append(np.array(z)[i:j])
    y = np.unique(np.array(y))
    return (x_traces,y,z_traces)

def read_df_to_xyz(df, xname, yname, zname, xrange='auto', yrange='auto'):
    if type(zname) == list:
        df_flat = []
        for i in range(len(zname)):
            df_flat.append(df[zname[i]].reset_index())
        x = df_flat[0][xname]
        y = df_flat[0][yname]
    else:
        df_flat = df[zname].reset_index()
        x = df_flat[xname]
        y = df_flat[yname]
    if type(zname) == list:
        z = []
        for i in range(len(zname)):
            z.append(df_flat[i][zname[i]])
    else:
        z = df_flat[zname]
    if (xrange == 'auto'):
        xrange = [0,len(x)] # use iloc instead of loc
    if (yrange == 'auto'):
        yrange = [0,len(y)] # use iloc instead of loc
    x = x[xrange[0]:xrange[1]]
    y = y[yrange[0]:yrange[1]]
    x = (np.unique(np.array(x)) if (x[0] < x[len(x)-1]) else np.flip(np.unique(np.array(x))))
    y = (np.unique(np.array(y)) if (y[0] < y[len(y)-1]) else np.flip(np.unique(np.array(y))))
    
    if (type(zname) == list):
        for i in range(len(z)):
            try:
                z[i] = np.reshape(np.array(z[i]),(len(y),len(x)))
            except ValueError:
                y = y[0:-1]
                z[i] = np.reshape(np.array(z[i])[0:len(x)*len(y)],(len(y),len(x)))
    else:
        try:
            z = np.reshape(np.array(z),(len(y),len(x)))
        except ValueError:
            y = y[0:-1]
            z = np.reshape(np.array(z)[0:len(x)*len(y)],(len(y),len(x)))
    return (x,y,z)

def auto_read_to_xyz(run_id:int, xz_traces:bool=False):
    """
    Determine x, y, and z axes automatically
    """
    dataset = {}
    dataset['dataset'] = load_by_id(run_id)
    try:
        dataset['df'] = dataset['dataset'].to_pandas_dataframe_dict()
    except:
        dataset['df'] = dataset['dataset'].get_data_as_pandas_dataframe()
    
    yaxis_name = get_axis_name(dataset['df'], index=0)
    dataset[yaxis_name] = []
    dataset['yaxis_name'] = yaxis_name
    
    xaxis_name = get_axis_name(dataset['df'], index=1)
    dataset[xaxis_name] = []
    dataset['xaxis_name'] = xaxis_name
    
    zaxis_names = []
    for zaxis_name in list(dataset['df'].keys()):
        dataset[zaxis_name] = []
        zaxis_names.append(zaxis_name)
    dataset['zaxis_names'] = zaxis_names
    
    for i in range(len(zaxis_names)):
        if (xz_traces == True):
            [dataset[xaxis_name], dataset[yaxis_name], dataset[zaxis_names[i]]] = read_df_to_xz_traces(
                dataset['df'], xname=xaxis_name, yname=yaxis_name, zname=zaxis_names[i]) 
        else:
            [dataset[xaxis_name], dataset[yaxis_name], dataset[zaxis_names[i]]] = read_df_to_xyz(
                dataset['df'], xname=xaxis_name, yname=yaxis_name, zname=zaxis_names[i])
    return dataset

def get_axis_name(df_dict, index:int=0):
    for k, v in df_dict.items():
        df = v
        break
    return df.index.names[index]

def get_nice_label(raw_label, instrument, variable, latex:bool=True):
    """
    raw_label has format instrument_variable_gatename, where
    instrument is e.g. qdac1
    variables is e.g. I or V
    gatename is a string, which might include numbers and underscores
    """
    label = raw_label.split(instrument+'_')[-1]
    #if (variable not in raw_label):
    #    raise ValueError("Check that variable is in raw_label")
    gate_name = label.split(variable+'_')[-1].replace('_','\ ')
    if (latex == True):
        label = r'$'+variable+'_{\mathrm{'+gate_name+'}}$'
    else:
        label = variable+' '+gate_name
    return label

def plot_quadrature(quadrature, data_labels:list, xname:str, yname:str, figsize, 
                    transpose:bool=False, derivative='none', rescale_ydata={},  
                    global_vrange:bool=False, xlim:list=None, ylim:list=None, 
                    remove_offsets_config={}, smooth:bool=False, save_path=None):
    fig = plt.figure(figsize=figsize)
    grids = gs.GridSpec(2, 1, height_ratios=[0.03, 1])
    axs = []
    caxs = [] 
    axs.append(fig.add_subplot(grids[1, 0]))
    caxs.append(fig.add_subplot(grids[0, 0]))

    for data_label in data_labels:
        Q = datasets[data_label][quadrature]
        dq_dy, dq_dx = np.gradient(Q)
        if (derivative == 'dq_dx'):
            Q = dq_dx
        elif (derivative == 'dq_dy'):
            Q = dq_dy
        if (data_label == data_labels[0]):
            global_vmin = Q.min()
            global_vmax = Q.max()
        elif (global_vmin > Q.min()):
            global_vmin = Q.min()
        elif (global_vmax < Q.max()):
            global_vmax = Q.max()
    #print("global min: ",global_vmin)
    #print("global max: ",global_vmax)
    
    global_xrange = [None, None]
    global_yrange = [None, None]
    for data_label in data_labels:
        xdata = datasets[data_label][xname]
        ydata = datasets[data_label][yname]
        if ('scale' in rescale_ydata.keys() and 'offset' in rescale_ydata.keys()):
            ydata = apply_linear_transformation(ydata, rescale_ydata['scale'], rescale_ydata['offset'])
        if ():
            global_xrange[0]
        try:
            Q = (np.transpose(datasets[data_label][quadrature]) 
                 if (transpose == True) else datasets[data_label][quadrature]) 
            dq_dy, dq_dx = np.gradient(Q)
            if (derivative == 'dq_dx'):
                Q = dq_dx
            elif (derivative == 'dq_dy'):
                Q = dq_dy
                
            if (smooth == True):
                w = int(0.05*len(Q[0]))
                Q = (scipy.signal.convolve2d(Q, np.ones(5, w), mode='symm')/w)[w//2:-w//2]
                xdata = xdata[w//2:-w//2]
                
            if ('axis' in remove_offsets_config.keys()):
                if ('istop' in remove_offsets_config.keys()):
                    Q = remove_offsets(Q, axis=remove_offsets_config['axis'], istop=remove_offsets_config['istop'])
                else:
                    Q = remove_offsets(Q, axis=remove_offsets_config['axis'])
            
            if (global_vrange == True):
                im0 = axs[0].pcolormesh(xdata, ydata, Q, shading='auto',
                    vmin=-0.04, vmax=0.04)
            else:
                im0 = axs[0].pcolormesh(xdata, ydata, Q, shading='auto')
        except:
            print("try failed")
            Q = (np.transpose(datasets[data_label][quadrature][:-1,:]) 
                 if (transpose == True) else datasets[data_label][quadrature]) 
            dq_dy, dq_dx = np.gradient(Q)
            if (derivative == 'dq_dx'):
                Q = dq_dx
            elif (derivative == 'dq_dy'):
                Q = dq_dy
                
            if (global_vrange == True):
                im0 = axs[0].pcolormesh(xdata, ydata, Q, shading='auto', 
                    vmin=-0.3, vmax=0.3)
            else:
                im0 = axs[0].pcolormesh(xdata, ydata, Q, shading='auto')
            if ('axis' in remove_offsets.keys()):
                if ('istop' in remove_offsets.keys()):
                    Q = remove_offsets(Q, axis=remove_offsets_config['axis'], istop=remove_offsets_config['istop'])
                else:
                    Q = remove_offsets(Q, axis=remove_offsets_config['axis'])

    cb0 = fig.colorbar(im0, cax=caxs[0], orientation='horizontal')
    caxs[0].set_title(quadrature+' (V)', fontsize=20)

    for i in range(1):
        axs[i].set_xlabel(xname+' (V)', fontsize=16)
        axs[i].set_ylabel(yname+' (V)', fontsize=16)
        if (xlim is not None):
            axs[i].set_xlim(xlim[0], xlim[1])
        if (ylim is not None):
            axs[i].set_ylim(ylim[0], ylim[1])
    plt.tight_layout()
    gc.collect()
    if (save_path is not None):
        plt.savefig(save_path, dpi=400)
        
    #if (click_to_zoom_in == True):
    #    xy = plt.ginput(n=1, show_clicks=True)
    #    print(xy)