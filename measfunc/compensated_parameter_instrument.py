from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.utils.validators import Numbers


class CompensatedParameterInstrument(Instrument):
    def __init__(self, name: str,
                 prime_parameter: Parameter,
                 compensating_parameter: Parameter,
                 *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)
        self.prime_parameter = prime_parameter
        self.compensating_parameter = compensating_parameter
        self.add_parameter('slope',
                           initial_value=0,
                           unit='a.u.',
                           label='Slope',
                           vals=Numbers(-100, 100),
                           get_cmd=None,
                           set_cmd=None)

        self.add_parameter('intercept',
                           initial_value=0,
                           unit='V',
                           label='Slope',
                           vals=Numbers(-1, 1),
                           get_cmd=None,
                           set_cmd=None)

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
        slope = self.root_instrument.slope()
        intercept = self.root_instrument.intercept()
        self.root_instrument.compensating_parameter(value*slope+intercept)

    def get_raw(self):
        return self.root_instrument.prime_parameter.get()
