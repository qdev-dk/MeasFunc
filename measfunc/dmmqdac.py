import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes import Parameter, ParameterWithSetpoints
from qcodes.utils.validators import Numbers, Arrays
from typing import Any, Iterable, Tuple, Union, Optional
from time import sleep



class BufferedAcquisitionController(Instrument):
    """
    Usage:
        buffered_acquisition_controller = BufferedAcquisitionController(name, dmm, qdac, **kwargs)
        zdata = dmm_acquisition()
    """
    def __init__(self, name: str, dmm: Instrument, qdac: Instrument,
                 fast_channel: Union[InstrumentChannel, Parameter], # does this also work with a custom qc.Parameter? if not replace with Union[InstrumentChannel, Parameter]
                 fast_vstart: float,
                 fast_vend: float,
                 fast_num_samples: int,
                 slow_channel: Union[InstrumentChannel, Parameter], # does this also work with a custom qc.Parameter? if not replace with Union[InstrumentChannel, Parameter]
                 slow_vstart: float,
                 slow_vend: float,
                 slow_num_samples: int,
                 fast_compensating_channel: Optional[InstrumentChannel],
                 fast_compensating_vstart: Optional[float],
                 fast_compensating_vend: Optional[float],
                 slow_compensating_channel: Optional[InstrumentChannel],
                 slow_compensating_vstart: Optional[float],
                 slow_compensating_vend: Optional[float],
                 *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.dmm = dmm
        self.qdac = qdac
        self.qdac_sync_source = 1

        fast_channel_setpoints = QDacChannelSetpoints(self, 'fast_channel_setpoints', fast_channel, fast_num_samples, fast_vstart, fast_vend)
        self.add_submodule('fast_channel_setpoints', fast_channel_setpoints)
        self.fast_indices = []
        self.fast_indices.append(self.get_qdac_channel_index(fast_channel))
        slow_channel_setpoints = QDacChannelSetpoints(self, 'slow_channel_setpoints', slow_channel, slow_num_samples, slow_vstart, slow_vend)
        self.add_submodule('slow_channel_setpoints', slow_channel_setpoints)
        self.slow_indices = []
        self.slow_indices.append(self.get_qdac_channel_index(slow_channel))

        if fast_compensating_channel:
            fast_compensating_channel_setpoints = QDacChannelSetpoints(self, 'fast_compensating_channel_setpoints', fast_compensating_channel, fast_num_samples, fast_compensating_vstart, fast_compensating_vend)
            self.add_submodule('fast_compensating_channel', fast_compensating_channel_setpoints)
            self.fast_indices.append(self.get_qdac_channel_index(fast_compensating_channel))
            # Also need to append fast_vstart, fast_stop? 
        if slow_compensating_channel:
            slow_compensating_channel_setpoints = QDacChannelSetpoints(self, 'slow_compensating_channel_setpoints', slow_compensating_channel, slow_num_samples, slow_compensating_vstart, slow_compensating_vend)
            self.add_submodule('slow_compensating_channel', slow_compensating_channel_setpoints)
            self.slow_indices.append(self.get_qdac_channel_index(slow_compensating_channel))
            # Also need to append slow_vstart, slow_stop? 

        self.add_parameter('buffered_2d_acquisition',
                           parameter_class=Buffered2DAcquisition,
                           qdac=qdac,
                           dmm=dmm,
                           vals=Arrays(shape=(self.fast_channel.num_samples.get_latest, # should this be the fast_channel_setpoints? 
                                              self.slow_channel.num_samples.get_latest)),
                           setpoints=(self.fast_channel.voltage_setpoints, self.slow_channel.voltage_setpoints)
                           )

        self.add_parameter('sample_rate',
                           initial_value=1/0.003,
                           unit='Hz',
                           label='Sample Rate',
                           vals=Numbers(1e-3,1),
                           get_cmd=None,
                           set_cmd=None)

    def ramp_voltages_2d_and_fetch(self):
        self.setup_dmm()
        self.sync_channels()
        fast_vstart = self.get_vstart_list('fast_channel')
        slow_vstart = self.get_vstart_list('slow_channel')
        fast_vend = self.get_vend_list('fast_channel')
        slow_vend = self.get_vend_list('slow_channel')
        step_length = 1/self.sample_rate()
        acquisition_time = self.qdac.ramp_voltages_2d(slow_chans=self.slow_indices,
                                                      slow_vstart=slow_vstart,
                                                      slow_vend=slow_vend,
                                                      fast_chans=self.fast_indices,
                                                      fast_vstart=fast_vstart,
                                                      fast_vend=fast_vend,
                                                      step_length=step_length,
                                                      slow_steps=self.slow_channel.n_points,
                                                      fast_steps=self.fast_channel.n_points)
        sleep(acquisition_time+0.1)
        data = self.dmm.fetch()
        self.dmm.display.clear()
        return np.array(data).reshape(self.slow_channel.nr_points(), self.fast_channel.nr_points())

    def setup_dmm(self):
        self._dmm.sample.count(self.slow_channel.n_points*self.fast_channel.n_points)
        self.t_sample = 1/self.root_instrument.sample_rate #0.003 # in seconds
        self._dmm.sample.timer(self.t_sample) 
        self.dmm.init_measurement()

    def sync_channels(self):
        for i in self.fast_indices + self.slow_indices[1:]:
            self.sync_channel(i)

    def sync_channel(self, i):
        qdac_channel = self.qdac.channels[i-1]
        qdac_channel.sync(self.qdac_sync_source)
        qdac_channel.sync_delay(0)
        qdac_channel.sync_duration(0.001)

    def get_vstart_list(self, channel_identifier:str):
        return [ch.vstart() for ch in self.channels if channel_identifier in ch.name]

    def get_vend_list(self, channel_identifier: str):
        return [ch.vend() for ch in self.channels if channel_identifier in ch.name]

    def get_qdac_channel_index(self, qdac_channel: Union[InstrumentChannel, Parameter]):
        try:
            channel_index = int(qdac_channel.name.split('ch_')[1]) # qdac's own channel 
        except IndexError:
            try:
                channel_index = int(qdac_channel.label.split('ch_')[1]) # custom channel, label indicates the channel
            except IndexError:
                raise ValueError("Invalid channel object. Needs to have channel number in name or label")
        return channel_index

class QDacChannelSetpoints(InstrumentChannel):
    """
    """
    def __init__(self, parent: Instrument, name: str, num_samples, vstart: float, vend: float,  **kwargs):
        super().__init__(parent, name, **kwargs)
        self.dim = name
        self.add_parameter('vstart',
                           initial_value=vstart,
                           unit='V',
                           label='V_'+self.dim+' start',
                           vals=Numbers(-10,10),
                           get_cmd=None,
                           set_cmd=None)

        self.add_parameter('vend',
                           initial_value=vend,
                           unit='V',
                           label='V_'+self.dim+' end',
                           vals=Numbers(-10,10),
                           get_cmd=None,
                           set_cmd=None)
        
        self.add_parameter('num_samples',
                           unit='',
                           initial_value=num_samples,
                           vals=Numbers(1,2e9),
                           get_cmd=None,
                           set_cmd=None)

        self.add_parameter('voltage_setpoints',
                           unit='V',
                           label='Voltage setpoints '+self.dim,
                           parameter_class=Setpoints,
                           vals=Arrays(shape=(self.num_samples.get_latest,)))
        self.voltage_setpoints.reset()

class Setpoints(Parameter):
    """
    A parameter that generates a setpoint array from vstart, vend and num points
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
        vstart = self.instrument.vstart.get()
        vend = self.instrument.vend.get()
        num_samples = self.instrument.num_samples.get() # n_points 
        self.sweep_array = np.linspace(vstart, vend, num_samples)

class Buffered2DAcquisition(ParameterWithSetpoints):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name)

    def get_raw(self):
        return self.Instrument.ramp_voltages_2d_and_fetch()

