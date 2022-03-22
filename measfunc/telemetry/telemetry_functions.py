import configparser
from collections import OrderedDict
from typing import Optional
import logging
import platform

from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.common.protocol import Envelope

import measfunc as ccp
import measfunc

telemetry_handler: Optional[AzureLogHandler] = None

LOGGING_SEPARATOR = ' ¦ '
""":data:`LOGGING_SEPARATOR` defines the str used to separate parts of the log
 message.
"""

FORMAT_STRING_DICT = OrderedDict([
    ('asctime', 's'),
    ('name', 's'),
    ('levelname', 's'),
    ('module', 's'),
    ('funcName', 's'),
    ('lineno', 'd'),
    ('message', 's')])
""":data:`FORMAT_STRING_DICT` defines the format used in logging messages.
"""


def get_formatter_for_telemetry() -> logging.Formatter:
    """
    Returns :class:`logging.Formatter` with only name, function name and
    message keywords from FORMAT_STRING_DICT
    """
    format_string_items = [f'%({name}){fmt}'
                           for name, fmt in FORMAT_STRING_DICT.items()
                           if name in ['message', 'name', 'funcName']]
    format_string = LOGGING_SEPARATOR.join(format_string_items)
    return logging.Formatter(format_string)


def flush_telemetry_traces() -> None:
    """
    Flush the traces of the telemetry logger. If telemetry is not enabled, this
    function does nothing.
    """
    global telemetry_handler
    if (ccp.telemetry_config['Telemetry'].getboolean('enabled')
            and telemetry_handler is not None):
        telemetry_handler.flush()


def start_telemetry() -> None:
    """
    Start telemetry, capturing all log messages and warnings and sending them
    to our Applications Insights cloud instance
    """

    global telemetry_handler

    instrumentation_key = ccp.telemetry_config['Telemetry']['instrumentation_key']

    root_logger = logging.getLogger(measfunc.__name__)
    root_logger.setLevel(logging.DEBUG)

    # remove previously set handlers
    for handler in (telemetry_handler,):
        if handler is not None:
            handler.close()
            root_logger.removeHandler(handler)

    # Transport module of opencensus-ext-azure logs info 'transmission
    # succeeded' which is also exported to azure if AzureLogHandler is
    # in root_logger. The following lines stops that.
    logging.getLogger('opencensus.ext.azure.common.transport').setLevel(
        logging.WARNING)

    def callback_function(envelope: Envelope) -> bool:
        envelope.tags['ai.user.accountId'] = platform.node()
        envelope.tags['ai.cloud.role'] = f"measfunc"
        return True

    telemetry_handler = AzureLogHandler(
        connection_string=f'InstrumentationKey={instrumentation_key}')
    telemetry_handler.add_telemetry_processor(callback_function)
    telemetry_handler.setLevel(logging.INFO)
    telemetry_handler.setFormatter(get_formatter_for_telemetry())
    root_logger.addHandler(telemetry_handler)
