import os 
import numpy as np 
import matplotlib 
import matplotlib.pyplot as plt 

def figure_settings():
    """
    These parameters are only set up once for all plots
    """
    plt.rc('font',family='Arial')
    plt.rc('font',size=12)
    plt.rc('axes',linewidth=1.5)
    plt.rc('xtick',direction='in')
    plt.rc('ytick',direction='in')
    plt.rc('xtick.major',size=6,pad=10,width=1.5)
    plt.rc('ytick.major',size=6,pad=10,width=1.5)
    plt.rc('xtick.minor',size=4,pad=5,width=1.0)
    plt.rc('ytick.minor',size=4,pad=5,width=1.0)
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams['mathtext.it'] = 'Arial:italic'
    params = {'mathtext.default':'it'}
    plt.rcParams.update(params)
# #
# #
def axes_settings(ax:matplotlib.axes.Axes, **ax_kwargs:dict):
    if ('colorbar' in ax_kwargs.keys()):
        cbar_kwargs = ax_kwargs['colorbar']
        #im0 = plt.cm.ScalarMappable(norm=plot_kwargs['norm'],cmap=plot_kwargs['cmap'])
        #im0._A = []
        cb = cbar_kwargs['fig'].colorbar(cbar_kwargs['im'], # 'ScalarMappable'
                                         cax=cbar_kwargs['ax'],
                                         orientation='horizontal')
        if 'label' in cbar_kwargs.keys():
            cb.ax.set_title(cbar_kwargs['label'])
    if 'xlabel' in ax_kwargs.keys():
        if 'xlabel_size' in ax_kwargs.keys():
            ax.set_xlabel(ax_kwargs['xlabel'],fontsize=ax_kwargs['xlabel_size'])
        else:
            ax.set_xlabel(ax_kwargs['xlabel'])
    if 'ylabel' in ax_kwargs.keys():
        if 'ylabel_size' in ax_kwargs.keys():
            ax.set_ylabel(ax_kwargs['ylabel'],fontsize=ax_kwargs['ylabel_size'])
        else:
            ax.set_ylabel(ax_kwargs['ylabel'])
    if 'xlim' in ax_kwargs.keys():
        if len(ax_kwargs['xlim']) == 2:
            ax.set_xlim(ax_kwargs['xlim'][0],ax_kwargs['xlim'][1])
    if 'ylim' in ax_kwargs.keys():
        if len(ax_kwargs['ylim']) == 2:
            ax.set_ylim(ax_kwargs['ylim'][0],ax_kwargs['ylim'][1])
    if 'xMajorTicks' in ax_kwargs.keys():
        ax.xaxis.set_major_locator(ax_kwargs['xMajorTicks'])
    if 'xMinorTicks' in ax_kwargs.keys():
        ax.xaxis.set_minor_locator(ax_kwargs['xMinorTicks'])
    if 'yMajorTicks' in ax_kwargs.keys():
        ax.yaxis.set_major_locator(ax_kwargs['yMajorTicks'])
    if 'yMinorTicks' in ax_kwargs.keys():
        ax.yaxis.set_minor_locator(ax_kwargs['yMinorTicks'])
    if 'tight_layout' in ax_kwargs.keys():
        if ax_kwargs['tight_layout']:
            plt.tight_layout()
    if 'savefig' in ax_kwargs.keys():
        try:
            if 'figname' in ax_kwargs['savefig'].keys():
                figname = ax_kwargs['savefig']['figname']
            else:
                figname = 'figure'
            if 'output_dir' in ax_kwargs['savefig'].keys():
                output_dir = ax_kwargs['savefig']['output_dir']
            else:
                output_dir = ''
            if 'format' in ax_kwargs['savefig'].keys():
                figformat = ax_kwargs['savefig']['format']
            else:
                figformat = 'png'
            if 'dpi' in ax_kwargs['savefig'].keys():
                dpi = ax_kwargs['savefig']['dpi']
            else:
                dpi = 400
            plt.savefig(os.path.join(output_dir,figname+'.'+figformat),dpi=dpi)
        except AttributeError:
            pass
    return ax
# #
# #
def get_requested_range(x, y, x_lims, y_lims=None, z=None):
    """
    Get the subarrays of data x (1D), y (1D) and z (2D) falling within the x
    and y ranges specified by lists x_lims = [x_min, x_max] and
    y_lims = [y_min, y_max]
    """
    x_idx = [np.searchsorted(x, x_lims[0]), np.searchsorted(x, x_lims[1])]
    xr = x[x_idx[0]:x_idx[1]]
    if y_lims is not None:
        y_idx = [np.searchsorted(y, y_lims[0]), np.searchsorted(y, y_lims[1])] 
        yr = y[y_idx[0]:y_idx[1]]
    else:
        yr = y[x_idx[0]:x_idx[1]]
    if ((z is not None) and (y_lims is not None)):
        zr = z[y_idx[0]:y_idx[1],x_idx[0]:x_idx[1]]
        return xr, yr, zr
    else:
        return xr, yr
# #
# #
def add_device_schematic_to_axis(ax, data_path:str, device_schematic_name:str):
    #sofias_path = os.path.join('F:\\', 'qcodes_local', 'Sofia') 
    #data_path = os.path.join(sofias_path, 'IMEC', 'Woodstar', 
    #    '9.3. T-gate JellyBean with SET', 'Die11_Subdie7_QBB36_1_3')
    with matplotlib.cbook.get_sample_data(os.path.join(data_path, "Device schematic", device_schematic_name)) as image_file:
        image = plt.imread(image_file)
    ax.imshow(image)
    ax.axis('off')
    return ax
    
    