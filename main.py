"""Apply pretrained (UK) DGMR model on real time French radar data.
- Makes a forecast with data from 2h ago.
- Compares the forecast with observation data.
- Plots a gif and an error chart.
"""

import datetime as dt

import numpy as np
import tensorflow as tf

from dgmr.data import get_input_array, get_list_files
from dgmr.model import predict
from dgmr.plot import plot_gif_forecast
from dgmr.settings import INPUT_IMG_SIZE, PLOT_PATH, RADAR_IMG_SIZE


def postprocessing(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Put back mask of radar field
    mask = np.tile(mask, (array.shape[0], 1, 1))
    # Get back to the full radar grid, that we cropped to fit in the neural network
    full_array = np.nan * np.ones(
        (array.shape[0], RADAR_IMG_SIZE[0], RADAR_IMG_SIZE[1])
    )
    full_array[:, : INPUT_IMG_SIZE[0], : INPUT_IMG_SIZE[1]] = array
    full_array = np.where(mask == 1, np.nan, full_array)
    return full_array


if __name__ == "__main__":

    date = dt.datetime.now(dt.timezone.utc)
    date = date - dt.timedelta(  # round date to 5 minutes
        minutes=date.minute % 5,
        seconds=date.second,
        microseconds=date.microsecond,
    )

    run_date = date - dt.timedelta(minutes=5)

    print(f"---> Making DGMR forecast for date {run_date}")

    file_paths = get_list_files(run_date)
    if not all([f.exists() for f in file_paths]):
        raise FileNotFoundError("Some radar files are not available")

    x_array, mask = get_input_array(file_paths)

    x_array = x_array[:, : INPUT_IMG_SIZE[0], : INPUT_IMG_SIZE[1], :]
    input_tensor = tf.convert_to_tensor(x_array, dtype=tf.float32)

    output = predict(input_tensor)[0]
    # TODO : make forecast on Corsica

    output = postprocessing(output, mask)

    dest_path = PLOT_PATH / run_date.strftime("%Y-%m-%d_%Hh%M.gif")
    plot_gif_forecast(output, run_date, dest_path)
