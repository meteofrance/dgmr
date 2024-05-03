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

from dgmr.plot import plot_error_per_leadtime, plot_gif
from dgmr.predict import predict_deepmind
from dgmr.settings import (
    HEXAGONE_DATA_PATH,
    INPUT_STEPS,
    PRED_STEPS,
    SIZE_IMG,
    TIMESTEP,
)


def get_list_files(date: dt.datetime) -> List[Path]:
    delta = dt.timedelta(minutes=TIMESTEP)
    dates = [date + i * delta for i in range(-INPUT_STEPS + 1, PRED_STEPS + 1)]
    filenames = [d.strftime("%Y%m%d%H%M.npz") for d in dates]
    return [HEXAGONE_DATA_PATH / f for f in filenames]


def load_x_y_array(paths: List[Path]) -> np.ndarray:
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
    return array[:4], array[4:]


if __name__ == "__main__":
    date = dt.datetime.now()
    date = date - dt.timedelta(  # round date to 5 minutes
        minutes=date.minute % TIMESTEP,
        seconds=date.second,
        microseconds=date.microsecond,
    )
    date = date - dt.timedelta(minutes=120)

    print(f"---> Making DGMR forecast for date {date}")

    file_paths = get_list_files(date)
    if not all([f.exists() for f in file_paths]):
        print("ERROR : some files are not available ! Exiting...")
        exit()

    x_array, y_array = load_x_y_array(file_paths)
    input_tensor = tf.convert_to_tensor(x_array, dtype=tf.float32)

    output = predict_deepmind(input_tensor)

    y_hat_array = output[0, 0].numpy()  # remove member and channel dims
    y_array = y_array[:, :, :, 0]

    # TODO : append x_array to plot

    plot_gif(y_hat_array, y_array, date)
    plot_error_per_leadtime(y_hat_array, y_array, date)
