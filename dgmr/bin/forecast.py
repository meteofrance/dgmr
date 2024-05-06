"""Apply pretrained (UK) DGMR model on real time French radar data.
- Makes a forecast with data from 2h ago.
- Compares the forecast with observation data.
- Plots a gif and an error chart.
"""

import datetime as dt
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

from dgmr.plot import plot_error_per_leadtime, plot_gif_comparison, plot_gif_forecast
from dgmr.predict import predict_deepmind
from dgmr.settings import (
    HEXAGONE_DATA_PATH,
    INPUT_STEPS,
    PRED_STEPS,
    SIZE_IMG,
    TIMESTEP,
)


def get_list_files_with_obs(date: dt.datetime) -> List[Path]:
    delta = dt.timedelta(minutes=TIMESTEP)
    dates = [date + i * delta for i in range(-INPUT_STEPS + 1, PRED_STEPS + 1)]
    filenames = [d.strftime("%Y%m%d%H%M.npz") for d in dates]
    return [HEXAGONE_DATA_PATH / f for f in filenames]


def get_list_files_without_obs(date: dt.datetime) -> List[Path]:
    delta = dt.timedelta(minutes=TIMESTEP)
    dates = [date + i * delta for i in range(-INPUT_STEPS + 1, 1)]
    filenames = [d.strftime("%Y%m%d%H%M.npz") for d in dates]
    return [HEXAGONE_DATA_PATH / f for f in filenames]


def load_arrays(paths: List[Path]) -> np.ndarray:
    arrays = [np.load(path)["arr_0"] for path in paths]
    array = np.stack(arrays)
    # Crop array to fit in neural network
    array = array[:, :, : SIZE_IMG[1]]
    # Remove negative values (outside radar field)
    array = np.where(array < 0, 0, array)
    # Conversion from mm 5min to mm/h
    array = array / 100 * 12
    # Add channel dims
    array = np.expand_dims(array, -1)
    return array


if __name__ == "__main__":

    date = dt.datetime.now()
    date = date - dt.timedelta(  # round date to 15 minutes
        minutes=date.minute % 15,
        seconds=date.second,
        microseconds=date.microsecond,
    )
    

    # Forecast 2h ago + comparison with obs

    run_date = date - dt.timedelta(minutes=120)

    print(f"---> Making DGMR forecast for date {run_date}")

    file_paths = get_list_files_with_obs(run_date)
    if not all([f.exists() for f in file_paths]):
        print("ERROR : some files are not available ! Exiting...")
        exit()

    arrays = load_arrays(file_paths)
    x_array, y_array = arrays[:INPUT_STEPS], arrays[INPUT_STEPS:]
    input_tensor = tf.convert_to_tensor(x_array, dtype=tf.float32)

    output = predict_deepmind(input_tensor)

    y_hat_array = output[0, 0].numpy()  # remove member and channel dims
    y_array = y_array[:, :, :, 0]
    x_array = x_array[:, :, :, 0]

    forecast, obs = np.concatenate([x_array, y_hat_array]), np.concatenate(
        [x_array, y_array]
    )
    plot_gif_comparison(forecast, obs, run_date)
    plot_error_per_leadtime(y_hat_array, y_array, run_date)


    # Forecast with latest data

    run_date = date - dt.timedelta(minutes=15)

    print(f"---> Making DGMR forecast for date {run_date}")

    file_paths = get_list_files_without_obs(run_date)
    print(file_paths, [f.exists() for f in file_paths])
    if not all([f.exists() for f in file_paths]):
        print("ERROR : some files are not available ! Exiting...")
        exit()

    x_array = load_arrays(file_paths)
    input_tensor = tf.convert_to_tensor(x_array, dtype=tf.float32)

    output = predict_deepmind(input_tensor)
    y_hat_array = output[0, 0].numpy()  # remove member and channel dims
    x_array = x_array[:, :, :, 0]

    forecast = np.concatenate([x_array, y_hat_array])
    plot_gif_forecast(forecast, run_date)
