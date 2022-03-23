import numpy as np 
from qcodes import Parameter, ParameterWithSetpoints
from qcodes.utils.validators import Numbers, Arrays

class QDacStepSetpoints(Parameter):
    # TODO: add init
    def get_raw(self): 
        slow_start = self._root_intrument.slow_start() 
        slow_stop = self._root_intrument.slow_stop()
        slow_num_samples = self._root_intrument.slow_num_samples()
        return np.linspace(slow_start, slow_stop, slow_num_samples)

class QDacRampSetpoints(Parameter):
    # TODO: add init
    def get_raw(self): 
        fast_start = self._root_intrument.fast_start() 
        fast_stop = self._root_intrument.fast_stop()
        fast_num_samples = self._root_intrument.fast_num_samples()
        return np.linspace(fast_start, fast_stop, fast_num_samples)

class AcquireDMMData(ParameterWithSetpoints):
    def __init__(self, name, dmm, qdac, *args, **kwargs):
        super().__init__(name=name) #name=(kwargs['name'] if ('name' in kwargs.keys()) else 'DMMAcquisition'),
                         #label=(kwargs['name'] if ('name' in kwargs.keys()) else 'DMM Acquisition'),
                         #docstring='Buffered acquisition with a Keysight DMM, while ramping QDac')
        self._dmm = dmm 
        self._qdac = qdac 

    def get_raw(self):
        """
        """

        if ((self._qdac_channel is None) or (self._vstart is None) or (self._vstop is None)
            or (self._num_samples is None) or (self._sample_rate is None) 
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
        return data 

class DMMAcquisition(Instrument):
    def __init__(self, name, dmm, qdac, *args, **kwargs):
        super().__init__(name=name)
        #self._start = None 
        #self._stop = None 
        #self._fast_channel_setpoints = QDacRampSetpoints()

        self.add_parameter('fast_start')

        self.add_parameter('fast_stop') 

        self.add_parameter('fast_num_points')  

        self.add_parameter('fast_channel_setpoints',
                           class=QDacRampSetpoints,
                           get_cmd=None)

    def update_setpoints(self):
        
        self.fast_channel_setpoints(self.fast_start(), self.fast_stop(), self.num_samples())

