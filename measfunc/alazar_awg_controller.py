import warnings
import numpy as np
from typing import Union, Tuple
from measfunc.AlazarAcquisitionController import AlazarAcquisitionController


class AlazarAWGController(AlazarAcquisitionController):
    def __init__(self, name, alazar_name: str, awg=None, **kwargs) -> None:
        if (awg is not None):  # TODO: replace awg, awg_run_command and awg_stop_command with a dictionary
            self.awg = awg
            if ('awg_run_command' in kwargs.keys()):
                self.awg_run_command = kwargs['awg_run_command']
                del kwargs['awg_run_command']
            else:
                self.awg_run_command = 'run'

            if ('awg_stop_command' in kwargs.keys()):
                self.awg_stop_command = kwargs['awg_stop_command']
                del kwargs['awg_stop_command']
            else:
                self.awg_stop_command = 'stop'
        else:
            self.awg = None
            warnings.warn("Controller not initialized with an AWG. Not able to acquire triggered data from pulsed/burst-mode waveforms.")
        super().__init__(name, alazar_name, **kwargs)

    def pre_acquire(self):
        if (self.awg is not None):
            try:
                getattr(self.awg, self.awg_run_command)()
            except Exception:
                warnings.warn("Could not start awg with self.awg."+self.awg_run_command+"()")

    def post_acquire(self) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        """
        if (self.awg is not None):
            try:
                # #
                # # EXPERIMENTAL; clean up
                if (hasattr(self.awg, 'get_state')):
                    if (self.awg.get_state() == 'Running'):
                        getattr(self.awg, self.awg_stop_command)()
                    else:
                        getattr(self.awg, self.awg_stop_command)()
            except Exception:
                warnings.warn("Could not stop awg with self.awg."+self.awg_stop_command+"()")

        return super().post_acquire()
