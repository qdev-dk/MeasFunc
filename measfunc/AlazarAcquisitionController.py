import inspect
import logging
import warnings
import numpy as np
import ctypes
from typing import Union, Tuple, Iterable
from qcodes import Parameter, ParameterWithSetpoints
from qcodes.instrument_drivers.AlazarTech.ATS import AcquisitionController
from qcodes.utils.validators import Arrays

logger = logging.getLogger(__name__)


class AlazarAcquisitionController(AcquisitionController):
    """
    Usage:
        alazar_card = AlazarTech_ATS9440(name='Alazar_ATS9440')

        acquisition_controller = AlazarAcquisitionController(alazar_name='Alazar_ATS9440',
                                                             name='acquisition_controller',
                                                             average_buffers=True,
                                                             average_records=False,
                                                             integrate_samples=False)

        acquisition_time = 50*1e-3
        samples_per_record = 2001 # will be rounded up to 2528 due to memory and sample rate constraints

        acquisition_kwargs = {'mode': alazar_card.mode(), # 'NPT', i.e. no-pretrigger data, by default
                              'records_per_buffer': 1,
                              'buffers_per_acquisition': 1,
                              'channel_selection': 'ABCD'}

        alazar_kwargs = {'channel_range': 0.2,
                         'timeout_ticks': 0} # set timeout_ticks to be positive non-zero value to start software trigger in the absence of hardware trigger
        acquisition_controller.setup_acquisition(acquisition_time, samples_per_record, acquisition_kwargs=acquisition_kwargs, alazar_kwargs=alazar_kwargs)

        data = acquisition_controller.trace_acquisition()

    TODO: - channels should not be setpoints, but rather a result should be returned for each channel (similar to MultiParameter)
          - add functionality for software demodulation (which was removed here)
          - add functionality for "filtering" (which was removed here), but please make it readable/explain what it is
    """
    def __init__(self, name, alazar_name: str, **kwargs) -> None:
        self.shape_info = {
            'average_buffers': None,
            'average_records': None,
            'integrate_samples': None}
        self.update_dictionary(self.shape_info, ignore_invalid=True,
                               delete_kwarg=True, kwargs=kwargs)

        super().__init__(name, alazar_name, **kwargs)

        self.acquisition_config = {}
        for acquisition_parameter in inspect.signature(self._get_alazar().acquire).parameters.keys():
            if (acquisition_parameter != 'acquisition_controller'):
                self.acquisition_config[acquisition_parameter] = None
        self.acquisition_time = None

        self.board_info = self._get_alazar().get_idn()
        self.num_channels_on_card = self._get_alazar().channels

        self.add_parameter('num_enabled_channels',
                           label='number of enabled channels',
                           get_cmd=self._get_num_enabled_channels)

        self.add_parameter('time_setpoints',
                           unit='s',
                           label='time setpoints',
                           parameter_class=TimeSetpoints,
                           vals=Arrays(shape=(self._get_alazar().samples_per_record.get,)))

        self.add_parameter('record_indices',
                           unit='a.u.',
                           label='record indices',
                           max_value_callable=self._get_alazar().records_per_buffer.get,
                           parameter_class=IndexSetpoints,
                           vals=Arrays(shape=(self._get_alazar().records_per_buffer.get,)))

        self.add_parameter('buffer_indices',
                           unit='a.u.',
                           label='buffer indices',
                           max_value_callable=self._get_alazar().buffers_per_acquisition.get,
                           parameter_class=IndexSetpoints,
                           vals=Arrays(shape=(self._get_alazar().buffers_per_acquisition.get,)))

        self.add_parameter('channel_indices',
                           unit='a.u.',
                           label='channel indices',
                           max_value_callable=self.num_enabled_channels.get,
                           parameter_class=IndexSetpoints,
                           vals=Arrays(shape=(self.num_enabled_channels.get,)))

        self._update_data_setpoints_and_vals()
        self.add_parameter(name='dataset_acquisition',
                           parameter_class=DatasetAcquisition,
                           vals=Arrays(shape=self.data_shape),
                           setpoints=self.data_setpoints
                           )

        # Hardware constants
        self._min_sample_step = self._get_alazar().samples_divisor
        self._min_num_samples = self._get_alazar().samples_per_record.vals._min_value
        # self._pretrigger_alignment = self._get_alazar().pretrigger_alignment
        self._valid_sample_rates = list(self._get_alazar().sample_rate.vals._values)
        indices_to_remove = [i_sr for i_sr, sr in enumerate(self._valid_sample_rates) if (type(sr) == str)]
        for ir in indices_to_remove[::-1]:
            self._valid_sample_rates.remove(self._valid_sample_rates[ir])
        self._min_sample_rate = min(self._valid_sample_rates)
        self._max_sample_rate = max(self._valid_sample_rates)
        self._max_num_samples = self.board_info["max_samples"]

    def do_acquisition(self) -> float:
        """
        """
        value: float = self._get_alazar().acquire(acquisition_controller=self,
                                                  **self.acquisition_config)
        return value

    def _get_num_enabled_channels(self):
        """
        """
        return self._get_alazar().get_num_channels(self._get_alazar().channel_selection.raw_value)

    def _update_data_setpoints_and_vals(self):
        setpoints = []
        vals_shape = []

        if self.num_enabled_channels.get() > 1:
            setpoints.append(self.channel_indices)
            vals_shape.append(self.num_enabled_channels.get)

        nr_buffers = self._get_alazar().buffers_per_acquisition()
        if (not self.shape_info['average_buffers']) and nr_buffers > 1:
            setpoints.append(self.buffer_indices)
            vals_shape.append(self._get_alazar().buffers_per_acquisition.get)

        nr_records = self._get_alazar().records_per_buffer()
        if (not self.shape_info['average_records']) and nr_records > 1:
            setpoints.append(self.record_indices)
            vals_shape.append(self._get_alazar().records_per_buffer.get)

        if not self.shape_info['integrate_samples']:
            setpoints.append(self.time_setpoints)
            vals_shape.append(self._get_alazar().samples_per_record.get)

        self.data_setpoints = tuple(setpoints)
        self.data_shape = tuple(vals_shape)

    def raw_samples_to_voltages(self, raw_samples, bits_per_sample: int,
                                voltage_range: float, unsigned: bool = True):
        """
        UNTESTED for all but 14 bits_per_sample
        If using other packing besides constants.PackDefault.PACK_DEFAULT (packing to 16-bits),
        replace this function.
        TODO: add behaviour for 8-bits_per_sample
        """
        if (bits_per_sample not in [8, 12, 14, 16]):
            warnings.warn("Unknown bits per sample. Not converting to voltages")
            return raw_samples
        if not unsigned:
            warnings.warn("Signed bit conversions not implemented. Not converting to voltages")
            return raw_samples

        if (bits_per_sample == 8):
            uchar_max = ctypes.c_uint8(-1).value  # equal to 255
            shifted_samples = np.right_shift(raw_samples, 0)
            code_zero = uchar_max/2.0
            code_range = uchar_max/2.0
        elif (bits_per_sample == 12) or (bits_per_sample == 14):
            shifted_samples = np.right_shift(raw_samples, 16 - bits_per_sample)
            code_zero = np.left_shift(1, bits_per_sample - 1) - 0.5
            code_range = np.left_shift(1, bits_per_sample - 1) - 0.5
        elif (bits_per_sample == 16):
            shifted_samples = np.right_shift(raw_samples, 0)
            ushrt_max = ctypes.c_uint16(-1).value  # equal to 65535
            code_zero = ushrt_max/2.0
            code_range = ushrt_max/2.0
        voltages = np.float64(voltage_range*(shifted_samples - code_zero)/code_range)
        return voltages

    def find_and_set_closest_sample_rate(self, sample_rate: Union[float, int]):
        sample_rate_dists = np.abs(np.array(self._valid_sample_rates) - sample_rate)
        closest_sample_rate = self._valid_sample_rates[list(sample_rate_dists).index(sample_rate_dists.min())]
        self._get_alazar().sample_rate(closest_sample_rate)
        return self._get_alazar().sample_rate()

    def find_closest_samples_per_record(self, samples_per_record: int):
        sample_remainder = samples_per_record % self._min_sample_step
        valid_samples_per_record = samples_per_record + (self._min_sample_step - sample_remainder) % self._min_sample_step
        valid_samples_per_record = int(max(valid_samples_per_record, self._min_num_samples))
        return valid_samples_per_record

    def find_and_set_compatible_acquisition_time_and_samples_per_record(self, sample_rate: float, samples_per_record: int):
        """
        """
        pass

    def find_and_set_compatible_sample_rate_and_samples_per_record(self, acquisition_time: float, samples_per_record: int):
        """
        """
        valid_samples_per_record = self.find_closest_samples_per_record(samples_per_record)
        t_sample = acquisition_time/valid_samples_per_record
        sample_rate = self.find_and_set_closest_sample_rate(1.0/t_sample)
        valid_samples_per_record = self.find_closest_samples_per_record(int(acquisition_time*sample_rate))
        self.acquisition_config['samples_per_record'] = valid_samples_per_record

    def update_dictionary(self, dictionary: dict,
                          ignore_invalid: bool = False,
                          delete_kwarg: bool = False, kwargs={}):
        """
        """
        kwargs_to_delete = []
        for dict_parameter, set_value in kwargs.items():
            if dict_parameter in dictionary.keys():
                dictionary[dict_parameter] = set_value
                if delete_kwarg:
                    kwargs_to_delete.append(dict_parameter)
            else:
                if ignore_invalid:
                    warnings.warn("Invalid argument"+dict_parameter+" for dictionary. Not setting")

        for k in kwargs_to_delete:
            del kwargs[k]

    def alazar_config(self, **kwargs):
        """
        """
        alazar = self._get_alazar()
        num_channels = alazar.channels

        with alazar.syncing():
            for alazar_parameter, set_value in kwargs.items():
                is_parameter = False
                if hasattr(alazar, alazar_parameter):
                    is_parameter = True
                    getattr(alazar, alazar_parameter)(set_value)
                else:
                    is_channel_parameter = False
                    for ch in range(num_channels):
                        if hasattr(alazar, alazar_parameter+'{:d}'.format(ch)):
                            is_channel_parameter = True
                            getattr(alazar, alazar_parameter+'{:d}'.format(ch))(set_value)
                if not (is_parameter or is_channel_parameter):
                    warnings.warn("\nTrying to set channel parameter "+alazar_parameter+'\n'+"which doesn't exist. Not setting!")

    def setup_acquisition(self, acquisition_time: float,
                          samples_per_record: int,
                          acquisition_kwargs: dict,
                          alazar_kwargs: dict):
        """
        All-in-one setup function. To run any of the acquisition methods, run this or equivalent setup before.

        acquisition_kwargs: Check self.acquisition_config keys or ATS.AlazarTech_ATS.acquire signature

        alazar_kwargs: Check specific ATS class (such as ATS9440) added parameters.
                       If channel parameters are called without a channel index, set_value is set for all channels.
        TODO: - is there a nicer way to provide acquisition_time and samples_per_record?
        """
        if ('sample_rate' in alazar_kwargs.keys()):
            raise ValueError("This function finds suitable sample rate automatically."
                             "If you want to set sample rate instead of acquisition time, setup without this function.")

        self.update_dictionary(self.acquisition_config,
                               kwargs=acquisition_kwargs)

        self.find_and_set_compatible_sample_rate_and_samples_per_record(acquisition_time,
                                                                        samples_per_record)

        for alazar_parameter, set_value in self.acquisition_config.items():
            if hasattr(self._get_alazar(), alazar_parameter) and (set_value is not None):
                try:
                    getattr(self._get_alazar(), alazar_parameter)(set_value)
                except Exception:
                    warnings.warn("Could not set parameter "+alazar_parameter+" with value "+set_value)

        self.acquisition_time = (1.0/self._get_alazar().sample_rate())*self._get_alazar().samples_per_record()

        self.alazar_config(**alazar_kwargs)
        self._update_data_setpoints_and_vals()
        self.dataset_acquisition.setpoints = tuple(self.data_setpoints)
        self.dataset_acquisition.vals = Arrays(shape=self.data_shape)

    def pre_start_capture(self) -> None:
        """
        """
        samples_per_record = self._get_alazar().samples_per_record()
        records_per_buffer = self._get_alazar().records_per_buffer()
        buffers_per_acquisition = self._get_alazar().buffers_per_acquisition()

        if ((samples_per_record != self.acquisition_config['samples_per_record']) or
                (records_per_buffer != self.acquisition_config['records_per_buffer']) or
                (buffers_per_acquisition != self.acquisition_config['buffers_per_acquisition'])):

            raise RuntimeError("Update alazar card state to match samples_per_record to acquisition_config."
                               "Run find_and_set_compatible_sample_rate_and_samples_per_record")

        if ((samples_per_record * records_per_buffer) > self._max_num_samples):
            raise RuntimeError("Trying to acquire {} samples in one buffer. Maximum"
                               " number of samples is {}".format((samples_per_record * records_per_buffer), self._max_num_samples))

        num_enabled_channels = self.num_enabled_channels()
        if self.shape_info['average_buffers']:
            self.buffer = np.zeros(samples_per_record*records_per_buffer*num_enabled_channels)
        else:
            self.buffer = np.zeros((buffers_per_acquisition, samples_per_record*records_per_buffer*num_enabled_channels))

    def pre_acquire(self):
        pass

    def handle_buffer(self, data: np.ndarray, buffernum: int = 0):
        """
        """
        if self.shape_info['average_buffers']:
            # NOTE: there was a minor bug here before. Adding without dividing by the number of buffers
            # changes the range of ints. You get clipping (integer -> integer mod max_16_bit_int)
            # when you try to cast to uint16 even after just two buffers.
            # Luckily, the driver divided by the number of buffers before converting, in post_acquire.
            # Also, the native type of self.buffer is not enforced, and it's a floating point number,
            # so you might lose precision but not clip the data. Even if self.buffer was unsigned integer
            # type, the code would only have been a problem if the size was 16-bit (up to 65_536).
            # For 32-bit integer (2_147_483_647) self.buffer, user would have to get very unlucky,
            # as the card memory is equal to the 32-bit max integer.
            self.buffer += data / self.acquisition_config['buffers_per_acquisition']
        else:
            self.buffer[buffernum] = data

    def post_acquire(self) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        """

        samples_per_record = self._get_alazar().samples_per_record()
        records_per_buffer = self._get_alazar().records_per_buffer()
        buffers_per_acquisition = self._get_alazar().buffers_per_acquisition()

        if self.shape_info['average_buffers']:
            number_of_buffers = 1
        else:
            number_of_buffers = buffers_per_acquisition
        # NOTE: there was a bug in reshape before. num_enabled_channels dimension was last, after samples_per_record.
        # When you enable more than one channels, the original ordering gives you "multiplexed" data, where
        # channel_data['A'] corresponds to data[0::num_enabled_channels], channel_data['B'] corresponds to data[1::num_enabled_channels], etc.
        # In that case, all channel_data elements have data from all channels.
        reshaped_buf = self.reshape_buffer(number_of_buffers,
                                           records_per_buffer,
                                           samples_per_record)

        channel_data = {}
        for i_ch, ch in enumerate(self._get_alazar().channel_selection()):
            channel_data[ch] = reshaped_buf[:, i_ch, :, :]
            channel_data[ch] = self.postprocess_channel_data(channel_data[ch],
                                                             channel_index=i_ch)
        return np.array([channel_data[ch] for ch in channel_data.keys()])

    def reshape_buffer(self, number_of_buffers,
                       records_per_buffer, samples_per_record):
        # bit_per_sample is not the right atribute to use in the if statment
        if (self.board_info['bits_per_sample'] == 14):
            return self.buffer.reshape(number_of_buffers,
                                       self.num_enabled_channels(),
                                       records_per_buffer, samples_per_record)
        elif (self.board_info['bits_per_sample'] == 12):
            return np.moveaxis(self.buffer.reshape(number_of_buffers,
                                                   records_per_buffer,
                                                   samples_per_record,
                                                   self.num_enabled_channels()), -1, 1)

    def postprocess_channel_data(self, channel_data: np.ndarray, channel_index: int):
        """
        Average over buffers, records, samples
        Convert integers to voltage values
        """
        (num_buffers_per_acquisition, num_records_per_buffer, num_samples_per_record) = channel_data.shape
        if ((num_buffers_per_acquisition != (self.acquisition_config['buffers_per_acquisition'] if not self.shape_info['average_buffers'] else 1))
                or (num_records_per_buffer != self.acquisition_config['records_per_buffer'])
                or (num_samples_per_record != self.acquisition_config['samples_per_record'])):
            raise ValueError("Data has an invalid shape. Check acquisition_config")

        if self.shape_info['average_records']:
            channel_data = np.uint16(np.mean(channel_data, axis=1, keepdims=True))
        else:
            channel_data = np.uint16(channel_data)

        if self.shape_info['integrate_samples']:
            averaged_channel_data = np.zeros((num_buffers_per_acquisition, num_records_per_buffer, 1))
            for i in range(num_buffers_per_acquisition):
                for j in range(num_records_per_buffer):
                    averaged_channel_data[i, j] = np.average(channel_data[i, j, :])
            channel_data = np.uint16(averaged_channel_data)

        (num_buffers_per_acquisition, num_records_per_buffer, num_samples_per_record) = channel_data.shape
        channel_data = self.raw_samples_to_voltages(raw_samples=channel_data,
                                                    bits_per_sample=self.board_info['bits_per_sample'],
                                                    voltage_range=getattr(self._get_alazar(), 'channel_range'+'{:d}'.format(channel_index+1))())
        return channel_data


class TimeSetpoints(Parameter):
    """
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_raw(self):
        acquisition_time = self.instrument.acquisition_time
        return np.linspace(0, acquisition_time, self.instrument._get_alazar().samples_per_record())


class IndexSetpoints(Parameter):
    """
    """
    def __init__(self, max_value_callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_value_callable = max_value_callable
        self.set_sweep_array_to_index_array()

    def set_raw(self, value: Iterable[Union[float, int]]) -> None:
        self.sweep_array = value

    def get_raw(self):
        max_index = self.max_value_callable()
        if len(self.sweep_array) != max_index:
            self.set_sweep_array_to_index_array()
        return self.sweep_array

    def set_sweep_array_to_index_array(self):
        max_index = self.max_value_callable()
        self.sweep_array = np.linspace(0, max_index - 1, max_index, dtype=int)


class DatasetAcquisition(ParameterWithSetpoints):
    """
    TODO: reshape shape_info automatically and rerun setup_acquisition before running do_acquisition?
    """
    def __init__(self, name, *args, **kwargs):
        super().__init__(name=name, *args, **kwargs)

    def get_raw(self):
        data = self.instrument.do_acquisition()
        return np.squeeze(data)
