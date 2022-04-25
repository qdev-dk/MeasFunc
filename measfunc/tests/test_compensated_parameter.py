from qcodes.tests.instrument_mocks import DummyInstrument, DummyChannel
from measfunc.compensated_parameter_instrument import CompensatedParameterInstrument

def test_CompensatedParameterInstrument():
    dac = DummyInstrument(name='dac')
    compensated_parameter_instrument = CompensatedParameterInstrument(name= 'test_compensation',
                                                                      prime_parameter=dac.dac1,
                                                                      compensating_parameter=dac.dac2)

    compensated_parameter_instrument.compensatedparameter(2)
    assert compensated_parameter_instrument.compensatedparameter() == 2
    assert compensated_parameter_instrument.compensatedparameter() == dac.dac1()

    compensated_parameter_instrument.ch1.slope(1)
    compensated_parameter_instrument.ch1.intercept(1)
    compensated_parameter_instrument.compensatedparameter(4)

    assert compensated_parameter_instrument.compensatedparameter() == 4
    assert compensated_parameter_instrument.compensatedparameter() == dac.dac1()

    assert compensated_parameter_instrument.ch1.compensating_parameter() == 5
    assert compensated_parameter_instrument.ch1.compensating_parameter() == dac.dac2()
test_CompensatedParameterInstrument()