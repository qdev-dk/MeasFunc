from qcodes.tests.instrument_mocks import DummyInstrument
from measfunc.compensated_parameter_instrument import CompensatedParameterInstrument


def test_compensate_by_one_parameter():
    dac = DummyInstrument(name='dac')
    compensated_parameter_instrument = CompensatedParameterInstrument(name='test_compensation',
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

def test_compensate_by_many_parameters():
    dac = DummyInstrument(name='dac')
    compensated_parameter_instrument = CompensatedParameterInstrument(name='test_compensation',
                                                                      prime_parameter=dac.dac1,
                                                                      compensating_parameter=(dac.dac2, dac.dac3))

    compensated_parameter_instrument.compensatedparameter(2)
    assert compensated_parameter_instrument.compensatedparameter() == 2
    assert compensated_parameter_instrument.compensatedparameter() == dac.dac1()

    compensated_parameter_instrument.ch1.slope(1)
    compensated_parameter_instrument.ch1.intercept(1)
    compensated_parameter_instrument.ch2.slope(2)
    compensated_parameter_instrument.ch2.intercept(2)
    compensated_parameter_instrument.compensatedparameter(4)

    assert compensated_parameter_instrument.compensatedparameter() == 4
    assert compensated_parameter_instrument.compensatedparameter() == dac.dac1()

    assert compensated_parameter_instrument.ch1.compensating_parameter() == 5
    assert compensated_parameter_instrument.ch1.compensating_parameter() == dac.dac2()
    assert compensated_parameter_instrument.ch2.compensating_parameter() == 10
    assert compensated_parameter_instrument.ch1.compensating_parameter() == dac.dac2()

