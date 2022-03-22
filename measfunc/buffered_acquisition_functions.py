import time 
import numpy as np 
from qcodes import Parameter, ParameterWithSetpoints
from qcodes import Measurement 
from qcodes.utils.validators import Numbers, Arrays

"""
This is the functionality, in functional programming, 
that I'd like to implement 
"""

def setup_dmm_for_buffered_acquisition(dmm, voltage_range:float=1.0):
    """
    """
    dmm.device_clear() # necessary after a timeout error
    dmm.reset() # can be commented out 
    dmm.display.text('buffering...') # Displays the text to speed up dmm commands 
    dmm.range(voltage_range) 
    dmm.aperture_mode('ON')
    dmm.NPLC(0.06)
    dmm.trigger.source('EXT') #before was IMM
    dmm.trigger.slope('POS')
    dmm.trigger.count(1)
    dmm.trigger.delay(0.0)
    dmm.sample.source('TIM')
    dmm.timeout(5)
    
    # # 
    # # Setup dmm 
    #dmm.sample.count(num_samples)
    #t_sample = 1/sample_rate #0.003 # in seconds
    #if (t_sample < 0.001):
    #    raise ValueError("Trying to ramp too fast. Limit is 1 kHz")
    #dmm.sample.timer(t_sample) # t_sample should be much larger than integration time (NPLC)
    #dmm.init_measurement() 
    
def setup_dmm_for_step_acquisition(dmm):
    dmm.display.clear()
    dmm.reset() 
    dmm.NPLC(0.06)

def get_qdac_channel_index(qdac_channel):
    try: 
        channel_index = int(qdac_channel.name.split('ch_')[1]) # qdac's own channel 
    except IndexError: 
        try:
            channel_index = int(qdac_channel.label.split('ch_')[1]) # custom channel, label indicates the channel
        except IndexError: 
            raise ValueError("Invalid channel object. Needs to have channel number in name or label")
    return channel_index 

def get_buffered_1d_acquisition(qdac, qdac_channel, vstart:float, vstop:float, num_samples:int, 
    sample_rate:float, dmm, qdac_sync_source:int=1, 
    compensating_channel=None, compensating_vstart=None, compensating_vstop=None):
    """
    Usage:
        setup_dmm_for_buffered_acquisition(dmm_rf)
        xdata = np.linspace(3.45, 3.55, 401)
        ydata = get_buffered_1d_acquisition(qdac=qdac_A, 
                                            qdac_channel=V_SET_plunger1_gate, 
                                            vstart=3.45, vstop=3.55, 
                                            num_samples=401, sample_rate=1/0.003, 
                                            dmm=dmm_rf)
    """
    # # 
    # # Check if we compensate 
    if (compensating_channel is not None):
        if ((compensating_vstart is None) or (compensating_vstop is None)):
            raise ValueError("Give start and stop for compensating channel")
        else: 
            compensate_fast_axis = True 
    else:
        compensate_fast_axis = False 
    # # 
    # # Setup dmm 
    dmm.sample.count(num_samples)
    t_sample = 1/sample_rate #0.003 # in seconds
    if (t_sample < 0.001):
        raise ValueError("Trying to ramp too fast. Limit is 1 kHz")
    dmm.sample.timer(t_sample) # t_sample should be much larger than integration time (NPLC)
    dmm.init_measurement() 

    # # 
    # # Determine channel index 
    channel_index = get_qdac_channel_index(qdac_channel)
    if (compensate_fast_axis):
        compensating_channel_index = get_qdac_channel_index(compensating_channel)
    
    # # 
    # # Sync channels 
    for qdac_channel in qdac.channels:
        if (int(qdac_channel._name.split("chan")[1]) == channel_index):
            qdac_channel.sync(qdac_sync_source)
            qdac_channel.sync_delay(0)
            qdac_channel.sync_duration(0.001) # 1 ms is the minimum trigger-on duration (QDac 1)
            
        if (compensate_fast_axis):
            if (int(qdac_channel._name.split("chan")[1]) == compensating_channel_index):
                qdac_channel.sync(qdac_sync_source)
                qdac_channel.sync_delay(0)
                qdac_channel.sync_duration(0.001) # 1 ms is the minimum trigger-on duration (QDac 1)

    # # 
    # # Start rampgs 
    acquisition_time = qdac.ramp_voltages_2d(slow_chans=[], 
                                             slow_vstart=[], 
                                             slow_vend=[],
                                             fast_chans=([channel_index, compensating_channel_index] if (compensate_fast_axis) else [channel_index]), 
                                             fast_vstart=([vstart, compensating_vstart] if (compensate_fast_axis) else [vstart]),
                                             fast_vend = ([vstop, compensating_vstop] if (compensate_fast_axis) else [vstop]), 
                                             step_length=t_sample,
                                             slow_steps=1, 
                                             fast_steps=num_samples)

    #time.sleep(acquisition_time)
    # # 
    # # Move data to PC 
    data = dmm.fetch()

    dmm.display.clear() # Returns display to its normal state
    #setup_dmm_for_step_acquisition(dmm) 
    return data     

def get_buffered_2d_acquisition(qdac, 
    slow_channel, slow_vstart:float, slow_vstop:float, slow_num_samples:int, 
    fast_channel, fast_vstart:float, fast_vstop:float, fast_num_samples:int, 
    sample_rate:float, dmm, qdac_sync_source:int=1, 
    slow_compensating_channel=None, slow_compensating_vstart=None, slow_compensating_vstop=None,
    fast_compensating_channel=None, fast_compensating_vstart=None, fast_compensating_vstop=None):
    """
    """
    # # 
    # # Check if we compensate 
    if (slow_compensating_channel is not None):
        if ((slow_compensating_vstart is None) or (slow_compensating_vstop is None)):
            raise ValueError("Give start and stop for slow-compensating channel")
        else:
            compensate_slow_axis = True 
    else:
        compensate_slow_axis = False 
        
    if (fast_compensating_channel is not None):
        if ((fast_compensating_vstart is None) or (fast_compensating_vstop is None)):
            raise ValueError("Give start and stop for fast-compensating channel")
        else: 
            compensate_fast_axis = True 
    else:
        compensate_fast_axis = False 
    
    #if ((slow_compensating_channel is not None) and ((slow_compensating_vstart is None) or (slow_compensating_vstop is None))):
    #    raise ValueError("Give start and stop for slow-compensating channel")
    #if ((fast_compensating_channel is not None) and ((fast_compensating_vstart is None) or (fast_compensating_vstop is None))):
    #    raise ValueError("Give start and stop for fast-compensating channel")
    
    # # 
    # # Setup dmm 
    dmm.sample.count(slow_num_samples*fast_num_samples)
    t_sample = 1/sample_rate #0.003 # in seconds
    if (t_sample < 0.001):
        raise ValueError("Trying to ramp too fast. Limit is 1 kHz")
    dmm.sample.timer(t_sample) # t_sample should be much larger than integration time (NPLC)
    dmm.init_measurement() 

    # # 
    # # Determine channel index 
    slow_channel_index = get_qdac_channel_index(slow_channel)
    fast_channel_index = get_qdac_channel_index(fast_channel)
    if (compensate_slow_axis):
        slow_compensating_channel_index = get_qdac_channel_index(slow_compensating_channel)
    if (compensate_fast_axis):
        fast_compensating_channel_index = get_qdac_channel_index(fast_compensating_channel)

    # # 
    # # Sync the involved channels 
    for qdac_channel in qdac.channels:
        if (int(qdac_channel._name.split("chan")[1]) == fast_channel_index):
            qdac_channel.sync(qdac_sync_source)
            qdac_channel.sync_delay(0)
            qdac_channel.sync_duration(0.001) # 1 ms is the minimum trigger-on duration (QDac 1)
            
        if (compensate_slow_axis):
            if (int(qdac_channel._name.split("chan")[1]) == slow_compensating_channel_index):
                qdac_channel.sync(qdac_sync_source)
                qdac_channel.sync_delay(0)
                qdac_channel.sync_duration(0.001) # 1 ms is the minimum trigger-on duration (QDac 1) 
                
        if (compensate_fast_axis):
            if (int(qdac_channel._name.split("chan")[1]) == fast_compensating_channel_index):
                qdac_channel.sync(qdac_sync_source)
                qdac_channel.sync_delay(0)
                qdac_channel.sync_duration(0.001) # 1 ms is the minimum trigger-on duration (QDac 1) 

    # # 
    # # Start ramps 
    acquisition_time = qdac.ramp_voltages_2d(slow_chans=([slow_channel_index, slow_compensating_channel_index] 
                                                         if (compensate_slow_axis) else [slow_channel_index]), 
                                             slow_vstart=([slow_vstart, slow_compensating_vstart] if (compensate_slow_axis) else [slow_vstart]), 
                                             slow_vend = ([slow_vstop, slow_compensating_vstop] if (compensate_slow_axis) else [slow_vstop]),
                                             
                                             fast_chans=([fast_channel_index, fast_compensating_channel_index] 
                                                          if (compensate_fast_axis) else [fast_channel_index]), 
                                             fast_vstart=([fast_vstart, fast_compensating_vstart] if (compensate_fast_axis) else [fast_vstart]),
                                             fast_vend = ([fast_vstop, fast_compensating_vstop] if (compensate_fast_axis) else [fast_vstop]), 
                                             step_length=t_sample,
                                             slow_steps=slow_num_samples, 
                                             fast_steps=fast_num_samples)

    #time.sleep(acquisition_time)
    # # 
    # # Move data to PC 
    data = dmm.fetch()
    dmm.display.clear() # Returns display to its normal state
    #setup_dmm_for_step_acquisition(dmm) 
    return data  
  
def get_buffered_2d_acquisition_with_retry_with_averaging(qdac,  
    slow_channel, slow_vstart, slow_vstop, slow_num_samples, 
    fast_channel, fast_vstart, fast_vstop, fast_num_samples, 
    num_averages, dmm, 
    slow_compensating_channel=None, slow_compensating_vstart=None, slow_compensating_vstop=None,
    fast_compensating_channel=None, fast_compensating_vstart=None, fast_compensating_vstop=None):
    
    setup_dmm_for_buffered_acquisition(dmm=dmm, voltage_range=1)
    
    z = np.zeros(slow_num_samples*fast_num_samples) 
    for i_average in range(num_averages):
        #print("average number ",i_average,"...")
        try:
            zdata = get_buffered_2d_acquisition(qdac=qdac, 
                                                slow_channel=slow_channel, 
                                                slow_vstart=slow_vstart, slow_vstop=slow_vstop, slow_num_samples=slow_num_samples, 
                                                fast_channel=fast_channel, 
                                                fast_vstart=fast_vstart, fast_vstop=fast_vstop, fast_num_samples=fast_num_samples, 
                                                sample_rate=1/0.003, dmm=dmm,
                                                slow_compensating_channel=slow_compensating_channel,
                        slow_compensating_vstart=slow_compensating_vstart, slow_compensating_vstop=slow_compensating_vstop,
                        fast_compensating_channel=fast_compensating_channel,
                        fast_compensating_vstart=fast_compensating_vstart, fast_compensating_vstop=fast_compensating_vstop)
        except: # Try again in case of VisaIOError
            try:
                setup_dmm_for_buffered_acquisition(dmm=dmm, voltage_range=1) # this starts by resetting the instrument
                time.sleep(1)
                zdata = get_buffered_2d_acquisition(qdac=qdac, 
                                                    slow_channel=slow_channel, 
                                                    slow_vstart=slow_vstart, slow_vstop=slow_vstop, slow_num_samples=slow_num_samples, 
                                                    fast_channel=fast_channel, 
                                                    fast_vstart=fast_vstart, fast_vstop=fast_vstop, fast_num_samples=fast_num_samples, 
                                                    sample_rate=1/0.003, dmm=dmm,
                                                    slow_compensating_channel=slow_compensating_channel,
                        slow_compensating_vstart=slow_compensating_vstart, slow_compensating_vstop=slow_compensating_vstop,
                        fast_compensating_channel=fast_compensating_channel,
                        fast_compensating_vstart=fast_compensating_vstart, fast_compensating_vstop=fast_compensating_vstop)
            except: # If you are unlucky, you get timeout twice in a row. Try once more 
                setup_dmm_for_buffered_acquisition(dmm=dmm, voltage_range=1)
                time.sleep(1)
                zdata = get_buffered_2d_acquisition(qdac=qdac, 
                                                    slow_channel=slow_channel, 
                                                    slow_vstart=slow_vstart, slow_vstop=slow_vstop, slow_num_samples=slow_num_samples, 
                                                    fast_channel=fast_channel, 
                                                    fast_vstart=fast_vstart, fast_vstop=fast_vstop, fast_num_samples=fast_num_samples, 
                                                    sample_rate=1/0.003, dmm=dmm,
                                                    slow_compensating_channel=slow_compensating_channel,
                        slow_compensating_vstart=slow_compensating_vstart, slow_compensating_vstop=slow_compensating_vstop,
                        fast_compensating_channel=fast_compensating_channel,
                        fast_compensating_vstart=fast_compensating_vstart, fast_compensating_vstop=fast_compensating_vstop)
        z += zdata/num_averages 
    return z 
    
def buffered_do2d_with_retry_with_averaging(qdac,  
    slow_channel, slow_vstart, slow_vstop, slow_num_samples, 
    fast_channel, fast_vstart, fast_vstop, fast_num_samples, 
    num_averages, dmm, zdata_name:str, zdata_unit:str='V', 
    slow_compensating_channel=None, slow_compensating_vstart=None, slow_compensating_vstop=None,
    fast_compensating_channel=None, fast_compensating_vstart=None, fast_compensating_vstop=None):
    """
    Wrapper for get_buffered_2d_acquisition_with_retry_with_averaging to save data 
    """
    # # 
    # # 1. Create measurement instance 
    meas = [Measurement()] #, Measurement()]
    slow_voltage_range = np.linspace(slow_vstart, slow_vstop, slow_num_samples)
    fast_voltage_range = np.linspace(fast_vstart, fast_vstop, fast_num_samples)
    meas[0].register_parameter(slow_channel)
    meas[0].register_parameter(fast_channel)
    meas[0].register_custom_parameter(zdata_name, zdata_name, zdata_unit, setpoints=(slow_channel, fast_channel,))

    # # 
    # # 2. Start measurement 
    with meas[0].run() as datasaver1: #meas[1].run() as datasaver2: 
        zdata = get_buffered_2d_acquisition_with_retry_with_averaging(qdac,  
            slow_channel, slow_vstart, slow_vstop, slow_num_samples, 
            fast_channel, fast_vstart, fast_vstop, fast_num_samples, 
            num_averages, dmm, 
            slow_compensating_channel=slow_compensating_channel, 
            slow_compensating_vstart=slow_compensating_vstart, 
            slow_compensating_vstop=slow_compensating_vstop,
            fast_compensating_channel=fast_compensating_channel, 
            fast_compensating_vstart=fast_compensating_vstart, 
            fast_compensating_vstop=fast_compensating_vstop)
  
        # 3. Save data 
        datasaver1.add_result((slow_channel, np.repeat(slow_voltage_range, fast_num_samples)),
                          (fast_channel, np.tile(fast_voltage_range, slow_num_samples)),
                          (zdata_name, zdata))