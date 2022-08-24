import qcodes.dataset.experiment_container as exc
import qcodes as qc
import numpy as np
from datetime import datetime
import os
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cmx
import copy
from operator import itemgetter
from typing import Union
import time

plt.rc('font', size=12)
plt.rc('axes', linewidth=1.5)
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')
plt.rc('xtick.major', size=6, pad=10, width=1.5)
plt.rc('ytick.major', size=6, pad=10, width=1.5)
plt.rc('xtick.minor', size=4, pad=5, width=1.0)
plt.rc('ytick.minor', size=4, pad=5, width=1.0)


class Leakage_matrix():
    def __init__(self, qdac, sample_name,
                 channels_to_measure: Union[str, list] = 'all',
                 gate_names: Union[dict, type(None)] = None):

        self.sample_name = sample_name
        self.qdac = qdac
        self.channel_names = {ch.name[-4:]: ch for ch in self.qdac.channels}
        if channels_to_measure == 'all':
            self.channels_to_measure = self.qdac.channels
        else:
            self.channels_to_measure = itemgetter(*[f'ch{channel:02d}' for channel in channels_to_measure])(self.channel_names)

        if type(gate_names) == dict:
            if len(gate_names) != len(self.channels_to_measure):
                raise Exception(f'Only provided {len(gate_names)} but {len(self.channels_to_measure)} channels set to be measured')
        self.gate_names = gate_names

        # set current range
        for channel in self.channels_to_measure:
            channel.measurement_range('low')

        self.voltage_indexes = {channel.name: index for index, channel in enumerate(self.channels_to_measure)}
    
    def measure_leakage_matrix(self,
                               voltage_difference,
                               mode='differential',
                               calculate='both',
                               nplc=1,
                               plot=True):

        voltages_start, current_start, array_of_currents = self._measure_currents(voltage_difference, mode, nplc)
        base_data = (voltages_start, current_start, array_of_currents)

        if mode == 'absolute':
            if calculate == 'resistance':
                return voltages_start/current_start
            elif calculate == 'conductance':
                return current_start/voltages_start
            elif calculate == 'both':
                return current_start/voltages_start, voltages_start/current_start
            else:
                raise Exception(f'calculate: {calculate} is not accepted, try "resistance", "conductance" or "both"')

        if calculate == 'resistance':
            resistance_matrix = self._calculate_leakage_matrix(array_of_currents, current_start, voltage_difference, mode='resistance')
        elif calculate == 'conductance':
            conductance_matrix = self._calculate_leakage_matrix(array_of_currents, current_start, voltage_difference, mode='conductance')
        elif calculate == 'both':
            resistance_matrix = self._calculate_leakage_matrix(array_of_currents, current_start, voltage_difference, mode='resistance')
            conductance_matrix = self._calculate_leakage_matrix(array_of_currents, current_start, voltage_difference, mode='conductance')
        else:
            raise Exception(f'calculate: {calculate} is not accepted, try "resistance", "conductance" or "both"')

        if plot:
            if calculate == 'resistance':
                self._plot_leakage_matrix(resistance_matrix,
                                            gate_names=self.gate_names,
                                            xvals_=[0.2, 0],
                                            yvals_=[0, 0.2],
                                            values='resistance')
                return resistance_matrix, base_data
            elif calculate == 'conductance':
                self._plot_leakage_matrix(conductance_matrix,
                                            gate_names=self.gate_names,
                                            xvals_=[0.2, 0],
                                            yvals_=[0, 0.2],
                                            values='conductance')
                return conductance_matrix, base_data
            elif calculate == 'both':
                self._plot_leakage_matrix(resistance_matrix,
                                            gate_names=self.gate_names,
                                            xvals_=[0.2, 0],
                                            yvals_=[0, 0.2],
                                            values='resistance')

                self._plot_leakage_matrix(conductance_matrix,
                                            gate_names=self.gate_names,
                                            xvals_=[0.2, 0],
                                            yvals_=[0, 0.2],
                                            values='conductance')
                return resistance_matrix, conductance_matrix, base_data


    def _measure_currents(self, voltage_difference, mode='differential', nplc=1):
        for channel in self.channels_to_measure:  # set nplc
            channel.measurement_nplc(nplc)

        time.sleep(1/50*nplc)
        voltages_start = np.array([channel.dc_constant_V() for channel in self.channels_to_measure])
        current_start = np.array([channel.read_current_A()[0] for channel in self.channels_to_measure])

        if mode == 'absolute':
            return voltages_start, current_start, None        

        array_of_currents = np.zeros(shape=(len(self.channels_to_measure), len(self.channels_to_measure)))
        i = 0
        for voltage_channel in self.channels_to_measure:
            #  set to start + diff
            voltage_channel.dc_constant_V(voltages_start[self.voltage_indexes[voltage_channel.name]] + voltage_difference)
            currents = []

            time.sleep(1/50*nplc)  #wait for NPLC
            for curr_channel in self.channels_to_measure:
                currents.append(curr_channel.read_current_A()[0])

            array_of_currents[i, :] = np.array(currents)
            i += 1

            #  return to start
            voltage_channel.dc_constant_V(voltages_start[self.voltage_indexes[voltage_channel.name]])

        # leakage_matrix = self._calculate_leakage_matrix(array_of_currents, current_start, voltage_difference, mode=calculate)

        return voltages_start, current_start, array_of_currents

    def _calculate_leakage_matrix(self, array_of_currents, current_start, voltage_difference, mode='resistance'):
        if mode == 'resistance':
            results_array = np.abs(voltage_difference)/np.abs(array_of_currents-current_start)

        elif mode == 'conductance':
            results_array = np.abs(array_of_currents-current_start)/np.abs(voltage_difference)
        else:
            raise Exception(f'mode: {mode} not recognised, try "resistance" or "conductance"')

        return results_array

        # for i,voltage_channel in enumerate(self.channels_to_measure):
        #     results_array[i, :] = abs((voltages_start[self.voltage_indexes[voltage_channel.name]] + voltage_difference))/np.abs(array_of_currents[i,:])


    def _save_leakage_matrix(self, folder, voltages_start, current_start, leakage_matrix, voltage_indexes, array_of_currents, voltage_difference):
        if not os.path.exists(folder):
            os.mkdir(folder)

        np.save(folder+'/starting_currents.npy', current_start)
        np.save(folder+'/starting_voltages.npy', voltages_start)
        np.save(folder+'/leakage_matrix.npy', leakage_matrix)
        np.save(folder+'/current_array.npy', array_of_currents)
        np.save(folder+'/voltage_difference.npy', voltage_difference)

        if self.gate_names is not None:
            with open(folder+'/gate_names.json', 'w') as file:
                file.write(json.dumps(self.gate_names))

        with open(folder+'/voltage_index.json', 'w') as file:
            file.write(json.dumps(voltage_indexes))

    def _load_leakage_matrix(self, folder):
        if not os.path.exists(folder):
            raise Exception(f'folder {folder} does not exist')

        current_start = np.load(folder+'/starting_currents.npy')
        voltages_start = np.load(folder+'/starting_voltages.npy')
        leakage_matrix = np.load(folder+'/leakage_matrix.npy')
        array_of_currents = np.load(folder+'/current_array.npy')
        voltage_difference = np.load(folder+'/voltage_difference.npy')

        with open(folder+'/voltage_index.json', 'w') as file:
            voltage_indexes = json.load(file)
        gate_names = None
        if os.path.exists(folder + '/gate_names.json'):
            with open(folder + '/gate_names.json', 'w') as file:
                gate_names = json.load(file)

        return voltages_start, current_start, leakage_matrix, voltage_indexes, array_of_currents, voltage_difference, gate_names

    def _plot_leakage_matrix(self, leakage_matrix,
                             gate_names: Union[dict, type(None)] = None,
                             xvals_=[0.2, 0],
                             yvals_=[0, 0.2],
                             values='resistance'):

        fig = plt.figure(figsize=(11.5, 12.0))
        grids = gs.GridSpec(2, 8, height_ratios=[0.03, 1.0])
        plot_axis = fig.add_subplot(grids[1, :])
        color_axis = fig.add_subplot(grids[0, 1:4])
        text_axis = fig.add_subplot(grids[0, 4:])
        text_axis.axis('off')

        lm = copy.deepcopy(leakage_matrix)
        lm = np.abs(lm)
        epsilon = np.abs(lm.max() - lm.min())*1e-6
        lm += epsilon

        if values == 'resistance':
            cmin = 0
            cmax = min(5e4, np.max(lm))

        elif values == 'conductance':
            lm /= (1/25812.807)  # get in units of G0
            cmin = 0
            cmax = np.max(lm)

        # Plot data
        im0 = plot_axis.pcolormesh(lm, shading='flat', edgecolor=cmx.gray(0.35), linewidth=1.5,
                                    vmin=cmin, vmax=cmax)

        if gate_names is None:
            gate_names_plot = [ch.name for ch in self.channels_to_measure]
        else:
            if leakage_matrix.shape[0]!=len(gate_names):
                raise Exception(f'{len(gate_names)} provided names does not match size of leakage matrix {leakage_matrix.shape}')

            fixed_gate_names = {}
            for key, value in gate_names.items():
                fixed_gate_names[f'{int(key):02d}'] = value
            gate_names_plot = itemgetter(*[ch.name[-2:] for ch in self.channels_to_measure])(fixed_gate_names)

        # Text ticks
        # Adjust tick density
        plot_axis.locator_params(axis="x", nbins=len(gate_names_plot))
        plot_axis.locator_params(axis="y", nbins=len(gate_names_plot))
        # Define and plot ticks
        xticks = gate_names_plot
        xticks = [xticks[i].replace('_', ' ') for i in range(len(xticks))]
        yticks = copy.deepcopy(xticks)
        # yticks.reverse()
        plot_axis.set_xticklabels(xticks, rotation='-90', fontsize=16)
        plot_axis.set_yticklabels(yticks, rotation='horizontal', fontsize=16)
        # Adjust tick position
        # dx = 0.15/len(gate_names)
        # dy = 0. 
        xoffset = matplotlib.transforms.ScaledTranslation(xvals_[0], xvals_[1], fig.dpi_scale_trans)
        yoffset = matplotlib.transforms.ScaledTranslation(yvals_[0], yvals_[1], fig.dpi_scale_trans)
        for label in plot_axis.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + xoffset)
        for label in plot_axis.yaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + yoffset)

        # Plot colorbar
        cb0 = fig.colorbar(im0, cax=color_axis, ticks = [cmin,(cmin+cmax)/2,cmax], orientation='horizontal')
        if values == 'resistance':
            color_axis.set_title(r'Resistance [$ k\Omega $]', fontsize=26)
            cb0.ax.set_xticklabels([int(cmin*1e-3),int((cmin+cmax)/2*1e-3),int(cmax*1e-3)])
        if values == 'conductance':
            color_axis.set_title(r'Conductance [$ G_0 $]', fontsize=26)
            cb0.ax.set_xticklabels([cmin,np.round((cmin+cmax)/2,1),np.round(cmax,1)])

        for t in cb0.ax.get_xticklabels():
            t.set_fontsize(18)

        plt.text(-0.035, 0.81, "Voltage", fontsize=26, transform=plt.gcf().transFigure)
        plt.text(0.92, 0.02, "Current", rotation=-90, fontsize=26, transform=plt.gcf().transFigure)

        plt.tight_layout()
        # plt.savefig(os.path.join(data_path, "2022-05-15_leakage-tests", "Plots", "leakage_matrix_at_25mK.pdf"), bbox_inches="tight")
        # plt.savefig(os.path.join(data_path, "2022-05-15_leakage-tests", "Plots", "leakage_matrix_at_25mK.png"), dpi=400, bbox_inches="tight")
