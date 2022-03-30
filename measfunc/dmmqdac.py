import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.instrument.channel import InstrumentChannel
from qcodes import Parameter, ParameterWithSetpoints
from qcodes.utils.validators import Numbers, Arrays
from typing import Any, Iterable, Tuple, Union, Optional
from time import sleep

class BufferedAcquisitionController(Instrument):
    """
    Meant to be used with a QDac I and a Keysight digital multimeter
    Usage:
        buffered_acquisition_controller = BufferedAcquisitionController(name, dmm, qdac, **kwargs)
        zdata = buffered_acquisition_controller()
    """
    def __init__(self, name:str, dmm:Instrument, qdac:Instrument,
                 fast_channel:Union[InstrumentChannel, Parameter], 
                 fast_vstart:float,
                 fast_vend:float,
                 fast_num_samples:int,
                 slow_channel:Optional[Union[InstrumentChannel, Parameter]], 
                 slow_vstart:Optional[float],
                 slow_vend:Optional[float],
                 slow_num_samples:Optional[int],
                 fast_compensating_channel:Optional[InstrumentChannel],
                 fast_compensating_vstart:Optional[float],
                 fast_compensating_vend:Optional[float],
                 slow_compensating_channel:Optional[InstrumentChannel],
                 slow_compensating_vstart:Optional[float],
                 slow_compensating_vend:Optional[float],
                 *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.dmm = dmm
        self.qdac = qdac
        self.qdac_sync_source = 1
        self.channels = []

        fast_channel_setpoints = QDacChannelSetpoints(self, 'fast_channel_setpoints', fast_vstart, fast_vend, fast_num_samples)
        self.add_submodule('fast_channel_setpoints', fast_channel_setpoints)
        self.fast_channel_indices = []
        self.fast_channel_indices.append(self.get_qdac_channel_index(fast_channel))
        self.channels.append(fast_channel_setpoints)

        if slow_channel:
            slow_channel_setpoints = QDacChannelSetpoints(self, 'slow_channel_setpoints', slow_vstart, slow_vend, slow_num_samples)
            self.add_submodule('slow_channel_setpoints', slow_channel_setpoints)
            self.slow_channel_indices = []
            self.slow_channel_indices.append(self.get_qdac_channel_index(slow_channel))
            self.channels.append(slow_channel_setpoints)

        if fast_compensating_channel:
            fast_compensating_channel_setpoints = QDacChannelSetpoints(self, 'fast_compensating_channel_setpoints', fast_compensating_vstart, fast_compensating_vend, fast_num_samples)
            self.add_submodule('fast_compensating_channel', fast_compensating_channel_setpoints)
            self.fast_channel_indices.append(self.get_qdac_channel_index(fast_compensating_channel))
            self.channels.append(fast_compensating_channel_setpoints)

        if slow_compensating_channel:
            slow_compensating_channel_setpoints = QDacChannelSetpoints(self, 'slow_compensating_channel_setpoints', slow_compensating_vstart, slow_compensating_vend, slow_num_samples)
            self.add_submodule('slow_compensating_channel', slow_compensating_channel_setpoints)
            self.slow_channel_indices.append(self.get_qdac_channel_index(slow_compensating_channel))
            self.channels.append(slow_compensating_channel_setpoints)

        self.add_parameter('buffered_1d_acquisition',
                           vals=Arrays(shape=(self.fast_channel_setpoints.num_samples,)),
                           setpoints=(self.fast_channel_setpoints.voltage_setpoints,),
                           parameter_class=Buffered1DAcquisition)

        self.add_parameter('buffered_2d_acquisition',
                           vals=Arrays(shape=(self.fast_channel_setpoints.num_samples, 
                                              self.slow_channel_setpoints.num_samples)),
                           setpoints=(self.fast_channel_setpoints.voltage_setpoints, 
                                      self.slow_channel_setpoints.voltage_setpoints),
                           parameter_class=Buffered2DAcquisition)

        self.add_parameter('sample_rate',
                           initial_value=1/0.003,
                           unit='Hz',
                           label='Sample Rate',
                           vals=Numbers(1,1e4),
                           get_cmd=None,
                           set_cmd=None)

    def ramp_voltages_and_fetch(self): 
        """
        get 1d trace
        """
        self.setup_dmm_memory_and_sample_rate()
        self.sync_channels()
        fast_vstart = self.get_vstart_list(channel_identifier='fast')
        fast_vend = self.get_vend_list(channel_identifier='fast')
        step_length = 1/self.sample_rate()
        acquisition_time = self.qdac.ramp_voltages_2d(slow_chans=[], 
                                                      slow_vstart=[], 
                                                      slow_vend=[],
                                                      fast_chans=self.fast_channel_indices, 
                                                      fast_vstart=fast_vstart,
                                                      fast_vend=fast_vend, 
                                                      step_length=step_length,
                                                      slow_steps=1, 
                                                      fast_steps=self.fast_channel_setpoints.num_samples)

        sleep(acquisition_time + 0.1)
        data = self.dmm.fetch()
        self.dmm.display.clear()
        return data     

    def ramp_voltages_2d_and_fetch(self):
        """
        get data equivalent to 2d matrix (needs to be reshaped without doNds)
        """
        if (not self.slow_channel_setpoints):
            raise ValueError("Slow channel needs to be set to use ramp_voltages_2d_and_fetch")
        self.setup_dmm_memory_and_sample_rate()
        self.sync_channels()
        fast_vstart = self.get_vstart_list(channel_identifier='fast')
        fast_vend = self.get_vend_list(channel_identifier='fast')
        slow_vstart = self.get_vstart_list(channel_identifier='slow')
        slow_vend = self.get_vend_list(channel_identifier='slow')
        step_length = 1/self.sample_rate()
        acquisition_time = self.qdac.ramp_voltages_2d(slow_chans=self.slow_channel_indices,
                                                      slow_vstart=slow_vstart,
                                                      slow_vend=slow_vend,
                                                      fast_chans=self.fast_channel_indices,
                                                      fast_vstart=fast_vstart,
                                                      fast_vend=fast_vend,
                                                      step_length=step_length,
                                                      slow_steps=self.slow_channel_setpoints.num_samples(),
                                                      fast_steps=self.fast_channel_setpoints.num_samples())
        sleep(acquisition_time + 0.1)
        data = self.dmm.fetch()
        self.dmm.display.clear()
        return np.array(data).reshape(self.slow_channel_setpoints.num_samples.get_latest(), 
                                      self.fast_channel_setpoints.num_samples.get_latest())

    def setup_dmm_memory_and_sample_rate(self):
        """
        Some settings need to be set correctly before running this
        See setup_dmm_for_buffered_acquisition
        """
        slow_num_samples = (self.slow_channel_setpoints.num_samples() if (self.slow_channel_setpoints) else 1)
        self.dmm.sample.count(slow_num_samples*self.fast_channel_setpoints.num_samples())
        self.t_sample = 1/self.root_instrument.sample_rate() #0.003 # in seconds
        self.dmm.sample.timer(self.t_sample) 
        self.dmm.init_measurement()

    def setup_dmm_for_buffered_acquisition(self, voltage_range:float=1.0, NPLC:float=0.06):
        """
        NPLC: number of power line cycles ()

        NPLC: 0.06 with sample_rate 1/0.003
        NPLC: 0.02 with sample_rate 1/0.001 # ? 
        NPLC: 0.2 with sample_rate 1/0.01
        """
        self.dmm.device_clear() # necessary after a timeout error
        self.dmm.reset() 
        self.dmm.display.text('buffering...') # Displays the text to speed up dmm commands 
        self.dmm.range(voltage_range)
        self.dmm.aperture_mode('ON')
        self.dmm.NPLC(NPLC)
        self.dmm.trigger.source('EXT') 
        self.dmm.trigger.slope('POS')
        self.dmm.trigger.count(1)
        self.dmm.trigger.delay(0.0)
        self.dmm.sample.source('TIM')
        self.dmm.timeout(5)
        
    def setup_dmm_for_step_acquisition(dmm, NPLC:float=0.06):
        dmm.display.clear()
        dmm.reset() # default settings, inc. internal trigger 
        dmm.NPLC(NPLC)

    def sync_channels(self):
        for i in self.fast_channel_indices + self.slow_channel_indices[1:]:
            self.sync_channel(i)

    def sync_channel(self, i:int):
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
            channel_index = int(qdac_channel.name.split('chan')[1]) # qdac's own channel (InstrumentChannel)
        except IndexError:
            try:
                channel_index = int(qdac_channel.label.split('ch_')[1]) # custom channel (Parameter), label indicates the channel
            except IndexError:
                raise ValueError("Invalid channel object. Needs to have channel number in name or label")
        return channel_index


class QDacChannelSetpoints(InstrumentChannel):
    """
    """
    def __init__(self, parent:Instrument, name:str, vstart:float, vend:float,  num_samples:int, **kwargs):
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
        num_samples = self.instrument.num_samples.get() # num_samples 
        self.sweep_array = np.linspace(vstart, vend, num_samples)


class Buffered1DAcquisition(ParameterWithSetpoints):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def get_raw(self):
        return self.instrument.ramp_voltages_1d_and_fetch()


class Buffered2DAcquisition(ParameterWithSetpoints):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def get_raw(self):
        return self.instrument.ramp_voltages_2d_and_fetch()

