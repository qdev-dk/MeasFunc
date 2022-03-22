import pytest

import measfunc as ccp


@pytest.fixture(scope="session", autouse=True)
def disable_telemetry():
    """
    We do not want the tests to send up telemetric information, even when run
    locally, so we disable that with this fixture.
    """

    original_state = ccp.telemetry_config['Telemetry']['enabled']

    try:
        ccp.telemetry_config['Telemetry']['enabled'] = "False"
        yield
    finally:
        ccp.telemetry_config['Telemetry']['enabled'] = original_state
