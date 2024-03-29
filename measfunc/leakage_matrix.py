import copy
import json
import os
import time
from datetime import datetime
from operator import itemgetter
from typing import Optional, Union

import matplotlib
import matplotlib.cm as cmx
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np

plt.rc("font", size=12)
plt.rc("axes", linewidth=1.5)
plt.rc("xtick", direction="in")
plt.rc("ytick", direction="in")
plt.rc("xtick.major", size=6, pad=10, width=1.5)
plt.rc("ytick.major", size=6, pad=10, width=1.5)
plt.rc("xtick.minor", size=4, pad=5, width=1.0)
plt.rc("ytick.minor", size=4, pad=5, width=1.0)


class Leakage_matrix:
    def __init__(
        self,
        qdac,
        sample_name,
        channels_to_measure: Union[str, list] = "all",
        gate_names: Optional[dict] = None,
    ):

        self.sample_name = sample_name
        self.qdac = qdac
        self.channel_names = {ch.name[-4:]: ch for ch in self.qdac.channels}
        if channels_to_measure == "all":
            self.channels_to_measure = self.qdac.channels
        else:
            self.channels_to_measure = itemgetter(
                *[f"ch{channel:02d}" for channel in channels_to_measure]
            )(self.channel_names)

        if isinstance(gate_names, dict) and len(gate_names) != len(
            self.channels_to_measure
        ):
            raise ValueError(
                f"Only provided {len(gate_names)} "
                f"but {len(self.channels_to_measure)} channels set to be measured"
            )
        self.gate_names = gate_names

        # set current range
        for channel in self.channels_to_measure:
            channel.measurement_range("low")

        self.voltage_indexes = {
            channel.name: index
            for index, channel in enumerate(self.channels_to_measure)
        }

    def measure_leakage_matrix(
        self,
        voltage_difference,
        mode="differential",
        calculate="both",
        nplc=1,
        repetitions=1,
        plot=True,
        save_folder=None,
    ):

        self.voltage_difference = voltage_difference
        self._measure_currents(mode, nplc, repetitions)

        if calculate == "resistance":
            self.resistance_matrix = self._calculate_leakage_matrix(mode="resistance")
        elif calculate == "conductance":
            self.conductance_matrix = self._calculate_leakage_matrix(mode="conductance")
        elif calculate == "both":
            self.resistance_matrix = self._calculate_leakage_matrix(mode="resistance")
            self.conductance_matrix = self._calculate_leakage_matrix(mode="conductance")
        else:
            raise ValueError(
                f"calculate: {calculate} is not accepted, "
                'try "resistance", "conductance" or "both"'
            )

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            now_folder = f"{save_folder}/" + datetime.now().strftime("%Y%m%d_%H_%M")
            print(f"saved to {now_folder}")
            self._save_leakage_matrix(now_folder)

        if plot:
            if calculate == "resistance":
                self.plot_leakage_matrix(
                    self.resistance_matrix,
                    gate_names=self.gate_names,
                    values="resistance",
                    save_folder=now_folder,
                )
            elif calculate == "conductance":
                self.plot_leakage_matrix(
                    self.conductance_matrix,
                    gate_names=self.gate_names,
                    values="conductance",
                    save_folder=now_folder,
                )
            elif calculate == "both":
                self.plot_leakage_matrix(
                    self.resistance_matrix,
                    gate_names=self.gate_names,
                    values="resistance",
                    save_folder=now_folder,
                )

                self.plot_leakage_matrix(
                    self.conductance_matrix,
                    gate_names=self.gate_names,
                    values="conductance",
                    save_folder=now_folder,
                )

    def _measure_currents(self, mode="differential", nplc=1, repetitions=1):
        for channel in self.channels_to_measure:  # set nplc
            channel.measurement_nplc(nplc)

        time.sleep(1 / 50 * nplc)
        self.voltages_start = np.array(
            [channel.dc_constant_V() for channel in self.channels_to_measure]
        )
        self.current_start = np.zeros((repetitions, len(self.channels_to_measure)))
        for i in range(repetitions):
            self.current_start[i, :] = np.array(
                [channel.read_current_A()[0] for channel in self.channels_to_measure]
            )

        if mode == "absolute":
            return self.voltages_start, self.current_start, None

        self.array_of_currents = np.zeros(
            shape=(
                repetitions,
                len(self.channels_to_measure),
                len(self.channels_to_measure),
            )
        )
        for voltage_index, voltage_channel in enumerate(self.channels_to_measure):
            #  set to start + diff
            voltage_channel.dc_constant_V(
                self.voltages_start[self.voltage_indexes[voltage_channel.name]]
                + self.voltage_difference
            )

            for repetition in range(repetitions):
                time.sleep(1 / 50 * nplc)  # wait for NPLC
                currents = [
                    curr_channel.read_current_A()[0]
                    for curr_channel in self.channels_to_measure
                ]

                self.array_of_currents[repetition, voltage_index, :] = np.array(
                    currents
                )

            #  return to start
            voltage_channel.dc_constant_V(
                self.voltages_start[self.voltage_indexes[voltage_channel.name]]
            )

        if repetitions != 1:
            self.averaged_array_of_currents = np.average(self.array_of_currents, axis=0)
            self.var_of_current = np.var(self.array_of_currents, axis=0)
            self.averaged_current_start = np.average(self.current_start, axis=0)
            self.var_of_current_start = np.var(self.current_start, axis=0)
        else:
            self.averaged_array_of_currents = np.squeeze(self.array_of_currents)
            self.var_of_current = None
            self.averaged_current_start = np.squeeze(self.current_start)
            self.var_of_current_start = None

    def _calculate_leakage_matrix(self, mode="resistance"):
        if mode == "resistance":
            results_array = self.voltage_difference / (
                self.averaged_array_of_currents - self.averaged_current_start
            )

        elif mode == "conductance":
            results_array = (
                self.averaged_array_of_currents - self.averaged_current_start
            ) / self.voltage_difference
        else:
            raise ValueError(
                f'mode: {mode} not recognised, try "resistance" or "conductance"'
            )

        return results_array

    def _save_leakage_matrix(self, folder):
        os.mkdir(folder)

        np.save(f"{folder}/starting_currents.npy", self.current_start)
        np.save(f"{folder}/starting_voltages.npy", self.voltages_start)
        if hasattr(self, "resistance_matrix"):
            np.save(f"{folder}/resistance_matrix.npy", self.resistance_matrix)
        if hasattr(self, "conductance_matrix"):
            np.save(f"{folder}/conductance_matrix.npy", self.conductance_matrix)
        np.save(f"{folder}/current_array.npy", self.array_of_currents)
        np.save(f"{folder}/voltage_difference.npy", self.voltage_difference)

        if self.gate_names is not None:
            with open(f"{folder}/gate_names.json", "w") as file:
                json.dump(self.gate_names, file)

        with open(f"{folder}/voltage_index.json", "w") as file:
            json.dump(self.voltage_indexes, file)

    def load_leakage_matrix(self, folder):
        if not os.path.exists(folder):
            raise NotADirectoryError(f"folder {folder} does not exist")

        self.current_start = np.load(f"{folder}/starting_currents.npy")
        self.voltages_start = np.load(f"{folder}/starting_voltages.npy")
        try:
            self.resistance_matrix = np.load(f"{folder}/resistance_matrix.npy")
            print("loaded resistance")
        except OSError:
            print("resistance not found")

        try:
            self.conductance_matrix = np.load(f"{folder}/conductance_matrix.npy")
            print("loaded conductance")
        except OSError:
            print("conductance not found")
        self.array_of_currents = np.load(f"{folder}/current_array.npy")
        self.voltage_difference = np.load(f"{folder}/voltage_difference.npy")

    def plot_leakage_matrix(
        self,
        leakage_matrix,
        gate_names: Optional[dict] = None,
        tick_move_vals=None,
        values="resistance",
        save_folder=None,
        cmin=None,
        cmax=None,
    ):
        if tick_move_vals is None:
            tick_move_vals = [0.2, 0.2]
        fig = plt.figure(figsize=(11.5, 12.0))
        grids = gs.GridSpec(2, 8, height_ratios=[0.03, 1.0])
        plot_axis = fig.add_subplot(grids[1, :])
        color_axis = fig.add_subplot(grids[0, 1:4])
        text_axis = fig.add_subplot(grids[0, 4:])
        text_axis.axis("off")

        lm = copy.deepcopy(leakage_matrix)
        lm = np.abs(lm)
        if values == "conductance":
            lm /= 1 / 25812.807  # get in units of G0
        # epsilon = np.abs(lm.max() - lm.min())*1e-6
        # lm += epsilon
        if cmin is None and cmax is None:
            if values == "resistance":
                cmin = 1e8
                cmax = np.min([10e9, np.max(lm)])
                if cmax < 1e8:
                    cmin = np.min(lm)
                    cmax = np.max(lm)
                cmax = min(cmax, 10e9)

            elif values == "conductance":
                cmin = 0
                cmax = np.max(lm)

        # Plot data
        im0 = plot_axis.pcolormesh(
            lm,
            shading="flat",
            edgecolor=cmx.gray(0.35),
            linewidth=1.5,
            vmin=cmin,
            vmax=cmax,
        )

        if gate_names is None:
            gate_names_plot = [ch.name for ch in self.channels_to_measure]
        else:
            if leakage_matrix.shape[0] != len(gate_names):
                raise ValueError(
                    f"{len(gate_names)} provided names does not match "
                    f"size of leakage matrix {leakage_matrix.shape}"
                )

            fixed_gate_names = {
                f"{int(key):02d}": value for (key, value) in gate_names.items()
            }

            gate_names_plot = itemgetter(
                *[ch.name[-2:] for ch in self.channels_to_measure]
            )(fixed_gate_names)

        # Text ticks
        # Adjust tick density
        plot_axis.locator_params(axis="x", nbins=len(gate_names_plot))
        plot_axis.locator_params(axis="y", nbins=len(gate_names_plot))
        # Define and plot ticks
        xticks = gate_names_plot
        xticks = [xticks[i].replace("_", " ") for i in range(len(xticks))]
        yticks = copy.deepcopy(xticks)
        # yticks.reverse()
        plot_axis.set_xticklabels(xticks, rotation="-90", fontsize=16)
        plot_axis.set_yticklabels(yticks, rotation="horizontal", fontsize=16)
        # Adjust tick position
        # dx = 0.15/len(gate_names)
        # dy = 0.
        xoffset = matplotlib.transforms.ScaledTranslation(
            tick_move_vals[0], 0, fig.dpi_scale_trans
        )
        yoffset = matplotlib.transforms.ScaledTranslation(
            0, tick_move_vals[1], fig.dpi_scale_trans
        )
        for label in plot_axis.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + xoffset)
        for label in plot_axis.yaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + yoffset)

        # Plot colorbar
        cb0 = fig.colorbar(
            im0,
            cax=color_axis,
            ticks=[cmin, (cmin + cmax) / 2, cmax],
            orientation="horizontal",
        )

        scale = self._get_scale(cmin, cmax)
        if values == "resistance":
            color_axis.set_title(rf"Resistance [$ {scale[1]}\Omega $]", fontsize=26)
            cb0.ax.set_xticklabels(
                [
                    f"<{cmin*scale[0]:.3f}",
                    f"{(cmin+cmax)/2*scale[0]:.3f}",
                    f">{cmax*scale[0]:.3f}",
                ]
            )
        if values == "conductance":
            color_axis.set_title(r"Conductance [$ G_0 $]", fontsize=26)
            cb0.ax.set_xticklabels(
                [f"<{cmin:.3f}", f"{(cmin+cmax)/2:.3f}", f">{cmax:.3f}"]
            )

        for t in cb0.ax.get_xticklabels():
            t.set_fontsize(18)

        plt.text(-0.035, 0.81, "Voltage", fontsize=26, transform=plt.gcf().transFigure)
        plt.text(
            0.92,
            0.02,
            "Current",
            rotation=-90,
            fontsize=26,
            transform=plt.gcf().transFigure,
        )

        plt.tight_layout()
        if save_folder is not None:
            plt.savefig(f"{save_folder}/{values}.pdf", bbox_inches="tight")
            plt.savefig(f"{save_folder}/{values}.png", dpi=400, bbox_inches="tight")

    def _get_scale(self, cmin, cmax):
        avg = (cmin + cmax) / 2
        if avg > 1e2 and avg < 1e5:
            return (1e-3, "k")
        if avg > 1e5 and avg < 1e7:
            return (1e-6, "M")
        if avg > 1e7:
            return (1e-9, "G")
