import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes import Parameter, ParameterWithSetpoints
from qcodes.utils.validators import Numbers, Arrays
from typing import Any, Iterable, Tuple, Union, Optional
from time import sleep


class DMMAcquisition(Instrument):
    def __init__(self, name: str, dmm: Instrument, qdac: Instrument,
                 fast: InstrumentChannel,
                 slow: InstrumentChannel,
                 fast_n_points: int,
                 slow_n_points: int,
                 fast_start: float,
                 fast_end: float,
                 slow_start: float,
                 slow_end: float,
                 fast_comp: Optional[InstrumentChannel],
                 slow_comp: Optional[InstrumentChannel],
                 fast_comp_start: float,
                 fast_comp_end: float,
                 slow_comp_start: float,
                 slow_comp_end: float,
                 *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.dmm = dmm
        self.qdac = qdac
        self.qdac_sync_source = 1

        fast_ch = QDacCh(self, 'fast', fast, fast_n_points, fast_start, fast_end)
        self.add_submodule('fast', fast_ch)
        self.fast_indexes = []
        self.fast_indexes.append(self.get_qdac_channel_index(fast))
        slow_ch = QDacCh(self, 'slow', slow, slow_n_points, slow_start, slow_end)
        self.add_submodule('slow', slow_ch)
        self.slow_indexes = []
        self.slow_indexes.append(self.get_qdac_channel_index(slow))

        if fast_comp:
            fast_comp_ch = QDacCh(self, 'fast_comp', fast_comp, fast_n_points, fast_comp_start, fast_comp_end)
            self.add_submodule('fast_comp', fast_comp_ch)
            self.fast_indexes.append(self.get_qdac_channel_index(fast_comp))
        if slow_comp:
            slow_comp_ch = QDacCh(self, 'slow_comp', slow_comp, slow_n_points, slow_comp_start, slow_comp_end)
            self.add_submodule('slow_comp', slow_comp_ch)
            self.slow_indexes.append(self.get_qdac_channel_index(slow_comp))


        self.add_parameter('dmm2dbuff',
                           parameter_class=AcquireDMMData,
                           qdac=qdac,
                           dmm=dmm,
                           vals=Arrays(shape=(self.fast.n_points.get_latest,
                                              self.slow.n_points.get_latest)),
                           setpoints=(self.fast.V_axis,self.slow.V_axis)
                           )

        self.add_parameter('sample_rate',
                           initial_value=1/0.003,
                           unit='Hz',
                           label='Sample Rate',
                           vals=Numbers(1e-3,1),
                           get_cmd=None,
                           set_cmd=None)

    def ramp_and_fetch(self):
        self.setup_dmm()
        self.sync_channels()
        fast_vstart = self.get_vstart_list('fast')
        slow_vstart = self.get_vstart_list('slow')
        fast_vend = self.get_vend_list('fast')
        slow_vend = self.get_vend_list('slow')
        step_length = 1/self.sample_rate()
        acquisition_time = self.qdac.ramp_voltages_2d(slow_chans=self.slow_indexes,
                                                      slow_vstart=slow_vstart,
                                                      slow_vend=self.slow_vend,
                                                      fast_chans=self.fast_indexes,
                                                      fast_vstart=fast_vstart,
                                                      fast_vend=fast_vend,
                                                      step_length=step_length,
                                                      slow_steps=self.slow.n_points,
                                                      fast_steps=self.fast.n_points)
        sleep(acquisition_time+0.1)
        data = self.dmm.fetch()
        self.dmm.display.clear()
        return np.array(data).reshape(self.fast.nr_points(),self.slow.nr_points())

    def setup_dmm(self):
        self._dmm.sample.count(slow_num_samples*fast_num_samples)
        self.t_sample = 1/self.root_instrument.sample_rate #0.003 # in seconds
        self._dmm.sample.timer(self.t_sample) 
        self.dmm.init_measurement()

    def sync_channels(self):
        for i in self.fast_indexes + self.slow_indexes[1:]:
            self.sync_channel(i)

    def sync_channel(self, i):
        qdac_channel = self.qdac.channels[i-1]
        qdac_channel.sync(self.qdac_sync_source)
        qdac_channel.sync_delay(0)
        qdac_channel.sync_duration(0.001)

    def get_vstart_list(self,part_name: str):
        return [ch.V_start() for ch self.channels if part_name in ch.name]
    def get_vend_list(self,part_name: str):
        return [ch.V_stop() for ch self.channels if part_name in ch.name]

    def get_qdac_channel_index(self, qdac_channel: InstrumentChannel):
        try:
            channel_index = int(qdac_channel.name.split('ch_')[1]) # qdac's own channel 
        except IndexError:
            try:
                channel_index = int(qdac_channel.label.split('ch_')[1]) # custom channel, label indicates the channel
            except IndexError:
                raise ValueError("Invalid channel object. Needs to have channel number in name or label")
        return channel_index

class QDacCh(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str, n_points, start: float, end: float,  **kwargs):
        super().__init__(parent, name, **kwargs)
        self.dim = name
        self.add_parameter('V_start',
                           initial_value=start,
                           unit='V',
                           label='V_'+self.dim+' start',
                           vals=Numbers(-1,1),
                           get_cmd=None,
                           set_cmd=None)

        self.add_parameter('V_stop',
                           initial_value=end,
                           unit='V',
                           label='V_'+self.dim+' stop',
                           vals=Numbers(-1,1),
                           get_cmd=None,
                           set_cmd=None)
        
        self.add_parameter('n_points',
                           unit='',
                           initial_value=n_points,
                           vals=Numbers(1,2e9),
                           get_cmd=None,
                           set_cmd=None)

        self.add_parameter('V_axis',
                           unit='V',
                           label='V Axis '+self.dim,
                           parameter_class=GeneratedSetPoints,
                           vals=Arrays(shape=(self.n_points.get_latest,)))
        self.V_axis.reset()

class GeneratedSetPoints(Parameter):
    """
    A parameter that generates a setpoint array from start, stop and num points
    parameters.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def set_raw(self, value: Iterable[Union[float, int]]) -> None:
        self.sweep_array = value

    def get_raw(self):
        return self.sweep_array

    def reset(self):
        V_start = self.instrument.V_start.get()
        V_stop = self.instrument.V_stop.get()
        nr = self.instrument.n_points.get()
        self.sweep_array = np.linspace(V_start, V_stop, nr)


class AcquireDMMData(ParameterWithSetpoints):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name)

    def get_raw(self):
        return self.Instrument.ramp_and_fetch()

