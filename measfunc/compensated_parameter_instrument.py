from qcodes import InstrumentChannel
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import Numbers
from typing import Union, Sequence


class CompensatedParameterInstrument(Instrument):
    def __init__(self, name: str,
                 prime_parameter: Parameter,
                 compensating_parameter: Union[Parameter, Sequence[Parameter]],
                 *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.prime_parameter = prime_parameter
        if type(compensating_parameter) == Parameter:
            compensating_parameter = (compensating_parameter,)

        for i, para in enumerate(compensating_parameter):
            ch = CompensatingChannel(self, f'ch{i+1}', para, 0, 0)
            self.add_submodule(f'ch{i+1}', ch)

        self.add_parameter('compensatedparameter',
                           initial_value=self.prime_parameter.get(),
                           unit=self.prime_parameter.unit,
                           label='Copensated '+self.prime_parameter.label,
                           vals=self.prime_parameter.vals,
                           parameter_class=CompensatedParameter)


class CompensatedParameter(Parameter):
    def __init__(self, name: str,
                 *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def set_raw(self, value):
        self.root_instrument.prime_parameter(value)
        channel_dict = self.root_instrument.submodules
        for ch in channel_dict.keys():
            comp_ch = channel_dict[ch]
            slope = comp_ch.slope()
            intercept = comp_ch.intercept()
            comp_ch.compensating_parameter(value*slope+intercept)

    def get_raw(self):
        return self.root_instrument.prime_parameter.get()


class CompensatingChannel(InstrumentChannel):
    def __init__(self, parent: Instrument,
                 name: str,
                 compensating_parameter: Parameter,
                 slope: float, intercept: float, **kwargs):
        super().__init__(parent, name, **kwargs)

        self.compensating_parameter = compensating_parameter

        self.add_parameter('slope',
                           initial_value=slope,
                           unit='a.u.',
                           label='Slope',
                           vals=Numbers(-100, 100),
                           get_cmd=None,
                           set_cmd=None)

        self.add_parameter('intercept',
                           initial_value=intercept,
                           unit='V',
                           label='Intercept',
                           vals=Numbers(-1, 1),
                           get_cmd=None,
                           set_cmd=None)
