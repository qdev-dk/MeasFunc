import os
from qcodes.dataset.plotting import plot_dataset


def export_data_and_png(data_obj, dir_path):
    dir_path_data = os.path.join(dir_path, "data")
    if not os.path.exists(dir_path_data):
        os.makedirs(dir_path_data)

    run_id = data_obj.run_id
    data_obj.to_pandas_dataframe().to_csv(os.path.join(dir_path_data, f"{run_id}.csv"))
    at = plot_dataset(data_obj)
    fig = at[0][0].figure
    fig.savefig(os.path.join(dir_path_data, f"{run_id}.png"))
