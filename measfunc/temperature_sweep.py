from triton import OxfordTriton
from qcodes.instrument_drivers.Lakeshore.Model_372 import Model_372
from typing import Tuple, NoReturn, Union, Callable, Optional
import numpy as np
from time import sleep
import logging

from qcodes.dataset import Measurement
from qcodes.parameters import Parameter, ParameterBase

##################

# Heater
_heater_range_curr = [1, 3.16, 10, 31.6]
_heater_range_temp = [0.03, 0.1, 0.3, 1]

# Magnet
critical_magnet_temp = 4.8

# Turbo
critical_temp = .78
critical_speed = 100

##################

logger = logging.getLogger('Temperature Sweep')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('./TemperatureSweepLog.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s \
                              - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def T1dMeasurement(
            fridge: OxfordTriton,
            lakeshore: Model_372,
            start: float,
            end: float,
            points: int,
            wait_time: float,
            *param_meas,
            pid: Tuple[float, float, float]=(10., 20., 0.),
            t_mc_ch: int=8,
            t_magnet_ch: int=13,
            magnet_active: bool=False,
            write_period: float=5.0,
            temperature_tolerance: float=2e-3,
            wait_cycle_time: float=0.5,
            wait_tolerance: float=0.1,
            wait_equilibration_time: float=1.5
):
    if not isinstance(t_mc_ch, int):
        raise TypeError('t_mc_ch must be an integer.')
    if not isinstance(t_magnet_ch, int):
        raise TypeError('t_magnet_ch must be an integer.')
    if not isinstance(points, int):
        raise TypeError('points must be an integer.')
    
    temp_setpoints = Parameter('temperature_setpoints', label='Temperature Setpoints', unit='K')
    sample_temp = Temperature('MC Temperature', fridge, lakeshore, t_mc_ch)
    setpoint_temp_diff = Parameter('temperature_deviation', label='Temperature Deviation', unit='K')

    turbo_state, heater_range = _init_sweep_state(
                                            fridge,
                                            lakeshore,
                                            t_mc_ch, 
                                            t_magnet_ch, 
                                            magnet_active, 
                                            pid,
                                            wait_cycle_time,
                                            wait_tolerance,
                                            wait_equilibration_time,
    )
    logger.debug(f'Initial turbo 1 state: {turbo_state}, \
                  and heater range: {heater_range}')
    
    meas = Measurement()
    meas.write_period = write_period
    meas.register_parameter(temp_setpoints)
    params = [sample_temp]
    meas.register_parameter(sample_temp, setpoints=(temp_setpoints,))
    meas.register_parameter(setpoint_temp_diff, setpoints=(temp_setpoints,))
    for param in param_meas:
        if isinstance(param, ParameterBase):
            params.append(param)
            meas.register_parameter(param, setpoints=(temp_setpoints,))

    with meas.run() as datasaver:

        setpoint_list = np.linspace(start, end, points)
        for setpoint in setpoint_list:
            if magnet_active == True:
                magnet_temperature = magnet_check(lakeshore, t_magnet_ch)
                logger.debug(f'Magnet check passed with magnet temperature \
                              T = {magnet_temperature} K')
            turbo_state, heater_range = _move_to_setpoint(
                                                        fridge,
                                                        lakeshore,
                                                        sample_temp,
                                                        setpoint, 
                                                        magnet_active, 
                                                        t_magnet_ch,
                                                        turbo_state,
                                                        heater_range,
                                                        temperature_tolerance,
            )
            params_get = [(param, param.get()) for param in params]
            datasaver.add_result(
                (temp_setpoints, setpoint), 
                (setpoint_temp_diff, setpoint - sample_temp()),
                *params_get
            )
            current_temperature = sample_temp()
            logger.debug(f'Continuing from measurement at {current_temperature} K')
            sleep(wait_time)

        return datasaver.dataset
    

def T2dMeasurement(
            fridge: OxfordTriton,
            lakeshore: Model_372,
            start_temp: float,
            end_temp: float,
            points_temp: int,
            wait_time_temp: float,
            parameter_fast: ParameterBase,
            start_fast: float,
            end_fast: float,
            points_fast: int,
            wait_time_fast: float,
            *param_meas,
            pid: Tuple[float, float, float] = (10., 20., 0.),
            t_mc_ch: int=8,
            t_magnet_ch: int=13,
            magnet_active: bool=False,
            write_period: float=5.0,
            temperature_tolerance: float=2e-3,
            wait_cycle_time: float=0.5,
            wait_tolerance: float=0.1,
            wait_equilibration_time: float=1.5
):
    if not isinstance(t_mc_ch, int):
        raise TypeError('t_mc_ch must be an integer.')
    if not isinstance(t_magnet_ch, int):
        raise TypeError('t_magnet_ch must be an integer.')
    if not isinstance(points_temp, int):
        raise TypeError('Number of temperature points must be an integer.')
    if not isinstance(points_fast, int):
        raise TypeError('Number of fast axis points must be an integer.')

    temp_setpoints = Parameter('temperature_setpoints', label='Temperature Setpoints', unit='K')
    sample_temp = Temperature('MC Temperature', fridge, lakeshore, t_mc_ch)
    setpoint_temp_diff = Parameter('temperature_deviation', label='Temperature Deviation', unit='K')
    
    turbo_state, heater_range = _init_sweep_state(
                                            fridge,
                                            lakeshore,
                                            t_mc_ch, 
                                            t_magnet_ch, 
                                            magnet_active, 
                                            pid,
                                            wait_cycle_time,
                                            wait_tolerance,
                                            wait_equilibration_time,
    )
    logger.debug(f'Initial turbo 1 state: {turbo_state}, \
                  and heater range: {heater_range}')
    
    meas = Measurement()
    meas.write_period = write_period
    meas.register_parameter(temp_setpoints)
    meas.register_parameter(parameter_fast)
    params = [sample_temp]
    meas.register_parameter(sample_temp, setpoints=(temp_setpoints, parameter_fast))
    meas.register_parameter(setpoint_temp_diff, setpoints=(temp_setpoints, parameter_fast))
    for param in param_meas:
        if isinstance(param, ParameterBase):
            params.append(param)
            meas.register_parameter(param, setpoints=(temp_setpoints, parameter_fast))

    with meas.run() as datasaver:

        temperature_setpoints = np.linspace(start_temp, end_temp, points_temp)
        fast_axis_setpoints = np.linspace(start_fast, end_fast, points_fast)

        for temperature_setpoint in temperature_setpoints:
            if magnet_active == True:
                magnet_temperature = magnet_check(lakeshore, t_magnet_ch)
                logger.debug(f'Magnet check passed with magnet temperature \
                 T = {magnet_temperature} K')

            turbo_state, heater_range = _move_to_setpoint(
                                                        fridge,
                                                        lakeshore,
                                                        sample_temp,
                                                        temperature_setpoint, 
                                                        magnet_active, 
                                                        t_magnet_ch,
                                                        turbo_state,
                                                        heater_range,
                                                        temperature_tolerance
            )

            current_temperature = sample_temp()
            logger.debug(f'Continuing to measurement at {current_temperature} K')
            for setpoint_fast in fast_axis_setpoints:
                parameter_fast(setpoint_fast)
                sleep(wait_time_fast)

                params_get = [(param, param.get()) for param in params]
                datasaver.add_result(
                    (temp_setpoints, temperature_setpoint), 
                    (parameter_fast, parameter_fast()),
                    (setpoint_temp_diff, temperature_setpoint - sample_temp()),
                    *params_get
                )
            
            sleep(wait_time_temp)

        return datasaver.dataset


class Temperature(Parameter):
    def __init__(
            self, 
            name: str, 
            fridge: OxfordTriton,
            lakeshore: Model_372,
            temperature_ch: int=8, 
            magnet_active: bool=False,
            t_magnet_ch: int=13,
            pid: Tuple[float, float, float] = (10., 20., 0.),
            wait_cycle_time: float=0.5,
            wait_tolerance: float=0.1,
            wait_equilibration_time: float=1.5,
            temperature_tolerance: float=2e-3,
            *args,
            **kwargs
    ):
        self.fridge = fridge
        self.lakeshore = lakeshore
        self.magnet_active = magnet_active
        self.ch_of_interest = temperature_ch
        self.t_magnet_ch = t_magnet_ch
        self.temperature_measurement = self.construct_ch(lakeshore)
        self.pid = pid
        self.turbo_state = fridge.turb1_state()
        self.heater_range = lakeshore.sample_heater.output_range()
        self.wait_cycle_time = wait_cycle_time
        self.wait_tolerance = wait_tolerance
        self.wait_equilibration_time = wait_equilibration_time
        self.temperature_tolerance = temperature_tolerance
        super().__init__(name=name.lower().replace(' ', '_'), unit='K', label=name, *args, **kwargs)

    def get_raw(self) -> float:
        return self.temperature_measurement()
    
    def set_raw(self, setpoint: float) -> None:
        self.heater_range = self.lakeshore.sample_heater.output_range()
        self.turbo_state = self.fridge.turb1_state()
        _move_to_setpoint(
            self.fridge,
            self.lakeshore,
            self.get_raw,
            setpoint, 
            self.magnet_active, 
            self.t_magnet_ch,
            self.turbo_state,
            self.heater_range,
            self.temperature_tolerance
        )

    def prepare_system(self) -> None:
        _init_sweep_state(
            self.fridge,
            self.lakeshore,
            self.ch_of_interest,
            self.t_magnet_ch,
            self.magnet_active,
            self.pid,
            self.wait_cycle_time,
            self.wait_tolerance,
            self.wait_equilibration_time,
        )

    def construct_ch(self, lakeshore: Model_372) -> Callable:
        temp_str = format_ch(self.ch_of_interest)
        return eval(f'lakeshore.ch{temp_str}.temperature')

def format_ch(temperature_ch: int) -> str:
    temp_str = '0' + str(temperature_ch)
    if len(temp_str) > 2:
        temp_str = temp_str[1:]
    return temp_str

def live_configurator(
    fridge: OxfordTriton,
    lakeshore: Model_372,
    sample_temp: Union[Parameter, Callable],
    future_setpoint: float, 
    heater_range: float,
    turbo_state: str,
    _heater_range_temp: list,
    _heater_range_curr: list,
) -> Tuple[str, float]:
    
    state = {1: 'on', -1: 'off'}
    if critical_temp == future_setpoint:
        target_state = 'off'
    else:
        target_state = state[np.sign(critical_temp - future_setpoint)]
    switch_turbo = (turbo_state != target_state)

    target_heater_range = _get_best_heater_range(
                                        _heater_range_temp, 
                                        _heater_range_curr, 
                                        future_setpoint
    )
    switch_range = (heater_range != target_heater_range)

    while switch_turbo or switch_range:
        
        if switch_turbo:
            live_state = state[np.sign(critical_temp - sample_temp())]
            turbo_state = _toggle_turbo(
                                    fridge,
                                    lakeshore,
                                    live_state, 
                                    turbo_state, 
                                    future_setpoint, 
                                    sample_temp, 
                                    critical_speed
            )
            switch_turbo = (turbo_state != target_state)

        if switch_range:
            live_range = _get_best_heater_range(
                                    _heater_range_temp, 
                                    _heater_range_curr, 
                                    sample_temp()
            )
            heater_range = _set_heater_range(lakeshore, sample_temp, live_range, heater_range)
            switch_range = (heater_range != target_heater_range)
        
        sleep(1)
    
    return turbo_state, heater_range


def _toggle_turbo(
        fridge: OxfordTriton, 
        lakeshore: Model_372,
        best_state: str, 
        turbo_state: str,
        future_setpoint: float,
        sample_temp: Union[Parameter, Callable],
        critical_speed: float,
) -> str:

    if turbo_state != best_state:

        fridge.turb1_state(best_state)
        logger.info('Turbo 1 has been switched ' + best_state + f' at T = {sample_temp()} K')

        if best_state == 'off':
            lakeshore.sample_heater.setpoint(sample_temp())
            
            while fridge.turb1_speed() > critical_speed:
                sleep(1)
            logger.debug(f'Continuing to setpoint {future_setpoint} K now. \
                          MC T = {sample_temp()} K and \
                          turbo 1 speed = {fridge.turb1_speed()} Hz.')
            lakeshore.sample_heater.setpoint(future_setpoint)
    
    return best_state


def _get_best_heater_range(
        _heater_range_temp: list, 
        _heater_range_curr: list,
        temperature: float,
) -> float:
    
    for temperature_threshold, current in zip(_heater_range_temp, _heater_range_curr):
        if temperature < temperature_threshold:
            break
    return current


def _set_heater_range(
        lakeshore: Model_372,
        sample_temp: Union[Parameter, Callable],
        best_range: float,
        heater_range: float,
) -> float:

    heater_dict = {1: '1mA', 3.16: '3.16mA', 10: '10mA', 31.6: '31.6mA'}

    if heater_range != best_range:
        lakeshore.sample_heater.output_range(heater_dict[best_range])
        logger.debug(f"Heater range changed to {best_range} mA at T = {sample_temp()} K.")
        return best_range
    return heater_range


def _move_to_setpoint(
        fridge: OxfordTriton,
        lakeshore: Model_372,
        sample_temp: Union[Parameter, Callable],
        setpoint: float, 
        magnet_active: bool, 
        t_magnet_ch: int,
        turbo_state: str,
        heater_range: float,
        temperature_tolerance: float
) -> Tuple[str, float]:

    if magnet_active == True:
        magnet_temperature = magnet_check(lakeshore, t_magnet_ch)
        logger.debug(f'Magnet check passed with magnet temperature \
                      T = {magnet_temperature} K')

    if heater_range == 'off':
        best_heater_range = _get_best_heater_range(_heater_range_temp, _heater_range_curr, sample_temp())
        _set_heater_range(lakeshore, sample_temp, best_heater_range, heater_range)

    lakeshore.sample_heater.setpoint(setpoint)
    logger.debug(f'New setpoint T = {setpoint} K')

    turbo_state, heater_range = live_configurator(
                                            fridge,
                                            lakeshore, 
                                            sample_temp, 
                                            setpoint, 
                                            heater_range, 
                                            turbo_state,
                                            _heater_range_temp,
                                            _heater_range_curr,
    )

    while not _stable(sample_temp, setpoint, temperature_tolerance):
        sleep(0.5)

    return turbo_state, heater_range


def magnet_check(
        lakeshore: Model_372, 
        t_magnet_ch: int,
) -> Union[NoReturn, float]:

    magnet_temp = eval(f'lakeshore.ch{t_magnet_ch}.temperature()')
    if magnet_temp > critical_magnet_temp:
        lakeshore.sample_heater.setpoint(0)
        logging.critical(f'Ended run due to hot magnet. T_magnet = {magnet_temp} K')
        raise ValueError(f'Magnet temperature is {magnet_temp} K. \
                         Cool down or set magnet_active=False and deactivate magnet.')
    return magnet_temp
    

def _set_pid_controller(
        lakeshore: Model_372, 
        pid: Tuple[float, float, float]
) -> None:
    
    P, I, D = pid
    lakeshore.sample_heater.P(P)
    lakeshore.sample_heater.I(I)
    lakeshore.sample_heater.D(D)
    logger.debug(f'PID-values set to: P = {P}, I = {I}, D = {D}')


def _set_active_channels(
                lakeshore: Model_372,
                t_mc_ch: int,
                t_magnet_ch: int,                
) -> None:

    for ch in lakeshore.channels:
        ch.enabled(False)
        logger.debug(f'Excitation on temperature channel {ch.short_name} \
                      is switched off')

    if t_mc_ch < 10:
        t_mc_ch_str = '0' + str(t_mc_ch)
    else:
        t_mc_ch_str = str(t_mc_ch)
    if t_magnet_ch < 10:
        t_magnet_ch_str = '0' + str(t_magnet_ch)
    else:
        t_magnet_ch_str = str(t_magnet_ch)

    keep_on_channels = [t_mc_ch_str, t_magnet_ch_str]
    keep_on_channels = ['ch%s' % ch for ch in keep_on_channels]
    for ch in keep_on_channels:
        eval(f'lakeshore.' + ch + '.enabled(True)')
        logger.debug(f'Excitation on temperature channel {ch} \
                      is switched on')


def _init_sweep_state(
                fridge: OxfordTriton,
                lakeshore: Model_372,
                t_mc_ch: int,
                t_magnet_ch: int,
                magnet_active: bool,
                pid: Tuple[float, float, float],
                wait_cycle_time: float,
                wait_tolerance: float,
                wait_equilibration_time: float,
) -> Tuple[str, float]:

    lakeshore.sample_heater.wait_cycle_time(wait_cycle_time)
    lakeshore.sample_heater.wait_tolerance(wait_tolerance)
    lakeshore.sample_heater.wait_equilibration_time(wait_equilibration_time)
    lakeshore.sample_heater.mode('closed_loop')

    if magnet_active:
        magnet_temperature = magnet_check(lakeshore, t_magnet_ch)
        logger.info(f'Magnet check passed with magnet temperature \
                     T = {magnet_temperature} K')

    if not magnet_active:
        t_magnet_ch = t_mc_ch

    _set_active_channels(lakeshore, t_mc_ch, t_magnet_ch)
    _set_pid_controller(lakeshore, pid)

    lakeshore.ch01.enabled(False)

    return fridge.turb1_state(), lakeshore.sample_heater.output_range()


def _stable(sample_temp: Union[Parameter, Callable], setpoint: float, tolerance: float=2e-3):
    diff = 0
    for _ in range(4):
        diff += abs(sample_temp() - setpoint)
        sleep(0.5)
    avg_diff = diff / 4
    if avg_diff < tolerance:
        return True
    else:
        return False