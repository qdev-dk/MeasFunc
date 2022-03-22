import time 
import numpy as np 
from qcodes import Parameter, ParameterWithSetpoints
from qcodes import Measurement 
from qcodes.utils.validators import Numbers, Arrays

"""
This is the functionality, in functional programming, 
that I'd like to implement (the trace acquisition case. 
Ideally the same class would be compatible with 1d and 2d acquisition.

This class works according to the api (see below) 
but is not compatible with doNds bc it lacks setpoints. 
"""

class BufferedAcquisition(Parameter):
    """
    Usage: 
        buffered_acquisition = DMMAcquisition(dmm=dmm_rf, qdac=qdac_A)
        buffered_acquisition.setup_dmm_for_buffered_acquisition() 
        buffered_acquisition.setup_buffered_1d_acquisition(qdac_channel=V_SET_plunger1_gate, vstart=3.45, vstop=3.55, 
            num_samples=401, sample_rate=1/0.003, qdac_sync_source=1)
        ydata = buffered_acquisition()
    """
    def __init__(self, dmm, qdac, **kwargs):
        super().__init__(name=(kwargs['name'] if ('name' in kwargs.keys()) else 'DMMAcquisition'),
            label=(kwargs['name'] if ('name' in kwargs.keys()) else 'DMM Acquisition'),
            docstring='Buffered acquisition with a Keysight DMM, while ramping QDac')
        self._dmm = dmm 
        self._qdac = qdac 
        self._qdac_channel = None 
        self.vstart = None 
        self.vstop = None 
        self.num_samples = None 
    # # 
    # # 
    def setup_dmm_for_buffered_acquisition(self, voltage_range:float=1):
        """
        """
        self._dmm.device_clear() # necessary after a timeout error
        self._dmm.reset() # can be commented out 
        self._dmm.display.text('buffering...') # Displays the text to speed up dmm commands 
        self._dmm.range(voltage_range) 
        self._dmm.aperture_mode('ON')
        self._dmm.NPLC(0.06)
        self._dmm.trigger.source('EXT') #before was IMM
        self._dmm.trigger.slope('POS')
        self._dmm.trigger.count(1)
        self._dmm.trigger.delay(0.0)
        self._dmm.sample.source('TIM')
        self._dmm.timeout(5)
    # # 
    # # 
    def setup_dmm_for_step_acquisition(self):
        """
        """
        self._dmm.display.clear()
        self._dmm.reset() 
        self._dmm.NPLC(0.06)
    # # 
    # # 
    def setup_buffered_1d_acquisition(self, qdac_channel, vstart:float, vstop:float, 
        num_samples:int, sample_rate:float, qdac_sync_source:int):
        """
        """
        self._qdac_channel = qdac_channel 
        self.vstart = vstart 
        self.vstop = vstop 
        self.num_samples = num_samples 
        self._sample_rate = sample_rate 
        self._qdac_sync_source = qdac_sync_source

        # # 
        # # Determine channel index 
        try: 
            self._channel_index = int(self._qdac_channel.name.split("ch_")[1]) # qdac's own channel 
        except IndexError: 
            try:
                self._channel_index = int(self._qdac_channel.label.split('ch_')[1]) # custom channel 
            except IndexError: 
                raise ValueError("Invalid channel object. Needs to have channel number in name or label")

        for qdac_channel in self._qdac.channels:
            if (int(qdac_channel._name.split("chan")[1]) == self._channel_index):
                qdac_channel.sync(qdac_sync_source)
                qdac_channel.sync_delay(0)
                qdac_channel.sync_duration(0.001) # 1 ms is the minimum trigger-on duration 

        self._dmm.sample.count(self.num_samples)
        self._t_sample = 1/self._sample_rate #0.003 # in seconds
        if (self._t_sample < 0.001):
            raise ValueError("Trying to ramp too fast. Limit is 1 kHz")
        self._dmm.sample.timer(self._t_sample) # t_sample should be much larger than integration time (NPLC)

        # # 
        # # Ready dmm for acquisition 
        self._dmm.init_measurement() 
    # # 
    # # 
    def get_raw(self):
        """
        """
        if ((self._qdac_channel is None) or (self.vstart is None) or (self.vstop is None)
            or (self.num_samples is None) or (self._sample_rate is None) 
            or (self._channel_index is None)):
            raise RuntimeError("Run setup_buffered_1d_acquisition first")

        # # 
        # # Acquire 
        acquisition_time = self._qdac.ramp_voltages_2d(slow_chans=[], 
                                                       slow_vstart=[], 
                                                       slow_vend=[],
                                                       fast_chans=[self._channel_index], 
                                                       fast_vstart=[self.vstart],
                                                       fast_vend=[self.vstop], 
                                                       step_length=self._t_sample,
                                                       slow_steps=1, 
                                                       fast_steps=self.num_samples)

        #time.sleep(acquisition_time)
        data = self._dmm.fetch()
        #dmm.display.clear() # Returns display to its normal state
        #setup_dmm_for_step_acquisition(dmm) 
        return data 