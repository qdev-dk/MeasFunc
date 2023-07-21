import Labber
from qcodes.dataset.data_set_protocol import DataSetProtocol


def qcodes_to_labber(data: DataSetProtocol) -> None:
    file_name = f"{data.sample_name}_{data.exp_name}_{data.captured_run_id}_{data.name}".replace(
        " ", "_"
    )
    xdata = data.to_xarray_dataset()
    lStep = [
        dict(name=dim, unit=xdata[dim].attrs["unit"], values=xdata[dim].values)
        for dim in reversed(list(xdata.dims))
    ]
    lLog = [
        dict(name=var, unit=xdata[var].attrs["unit"], vector=False)
        for var in xdata.data_vars
    ]
    f = Labber.createLogFile_ForData(file_name, lLog, lStep)
    if len(lStep) == 1:
        data = {vardim: xdata[vardim] for vardim in xdata.data_vars}
        f.addEntry(data)
    else:
        for i in range(len(lStep[1]["values"])):
            data = {vardim: xdata[vardim][i] for vardim in xdata.data_vars}
            f.addEntry(data)
