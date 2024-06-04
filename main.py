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

from dgmr.model import predict
from dgmr.plot import plot_gif_forecast
from dgmr.settings import (
    DATA_PATH,
    INPUT_IMG_SIZE,
    INPUT_STEPS,
    PLOT_PATH,
    RADAR_IMG_SIZE,
    TIMESTEP,
)


def get_list_files(date: dt.datetime) -> List[Path]:
    delta = dt.timedelta(minutes=TIMESTEP)
    dates = [date + i * delta for i in range(-INPUT_STEPS + 1, 1)]
    filenames = [d.strftime("%Y%m%d%H%M.npz") for d in dates]
    return [DATA_PATH / f for f in filenames]


def load_arrays(paths: List[Path]) -> np.ndarray:
    arrays = [np.load(path)["arr_0"] for path in paths]
    array = np.stack(arrays)
    # Crop array to fit in neural network
    array = array[:, :, : INPUT_IMG_SIZE[1]]
    # Remove negative values (outside radar field)
    mask = np.where(array[-1] < 0, 1, 0)
    array = np.where(array < 0, 0, array)
    # Conversion from mm 5min to mm/h
    array = array / 100 * 12
    # Add channel dims
    array = np.expand_dims(array, -1)
    return array, mask


def postprocessing(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Put back mask of radar field
    mask = np.tile(mask, (array.shape[0], 1, 1))
    array = np.where(mask == 1, np.nan, array)
    # Get back to the full radar grid, that we cropped to fit in the neural network
    full_array = np.nan * np.ones(
        (array.shape[0], RADAR_IMG_SIZE[0], RADAR_IMG_SIZE[1])
    )
    full_array[:, :, : INPUT_IMG_SIZE[1]] = array
    return full_array


if __name__ == "__main__":

    date = dt.datetime.now(dt.timezone.utc)
    date = date - dt.timedelta(  # round date to 15 minutes
        minutes=date.minute % 15,
        seconds=date.second,
        microseconds=date.microsecond,
    )

    run_date = date - dt.timedelta(minutes=15)
    run_date = dt.datetime(2024, 5, 31, 14)

    print(f"---> Making DGMR forecast for date {run_date}")

    file_paths = get_list_files(run_date)
    print([f.exists() for f in file_paths])
    if not all([f.exists() for f in file_paths]):
        print("ERROR : some files are not available ! Exiting...")
        exit()

    x_array, mask = load_arrays(file_paths)
    input_tensor = tf.convert_to_tensor(x_array, dtype=tf.float32)

    output = predict(input_tensor)[0]
    print(output.shape)

    output = postprocessing(output, mask)

    dest_path = PLOT_PATH / run_date.strftime("%Y-%m-%d_%Hh%M.gif")
    plot_gif_forecast(output, run_date, dest_path)
