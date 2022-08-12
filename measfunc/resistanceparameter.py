
from qcodes.instrument.parameter import Parameter

class Resistance(Parameter):
    def __init__(self, name, volt, current, unit='ohm'):
        # only name is required
        self.volt = volt
        self.current = current
        
        super().__init__(name, label='Ohm',
                         unit=unit)
        self._count = 0

    # you must provide a get method, a set method, or both.
    def get_raw(self):
        volt = self.volt.get()
        current = self.current.get()
        resistance = volt/current
        return resistance
