"""
Test that the configuration system works
"""
import measfunc as ccp


def test_config_sections_and_fields():
    assert ccp.telemetry_config.sections() == ['Telemetry']

    expected_keys = ['enabled', 'instrumentation_key']
    assert sorted(ccp.telemetry_config['Telemetry'].keys()) == expected_keys
