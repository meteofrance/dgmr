"""Applies pretrained (UK) DGMR model on real time French radar data."""

import datetime as dt

import numpy as np
import tensorflow as tf

from dgmr.data import get_input_array, get_list_files
from dgmr.model import predict, load_model
from dgmr.plot import plot_gif_forecast
from dgmr.settings import INPUT_IMG_SIZE, PLOT_PATH, RADAR_IMG_SIZE


def make_forecast(x_array:np.ndarray)->np.ndarray:
    """Makes a rainfall forecast on the full French radar domain."""

    size_y, size_x = INPUT_IMG_SIZE

    # The pretrained model has a predefined input/output shape, so we have to split the
    # domain in 2 parts and make 2 forecasts
    input_nw = x_array[:, : size_y, : size_x, :]
    tensor_nw = tf.convert_to_tensor(input_nw, dtype=tf.float32)
    input_se = x_array[:, -size_y:, -size_x:, :]
    tensor_se = tf.convert_to_tensor(input_se, dtype=tf.float32)

    model = load_model(INPUT_IMG_SIZE)

    print("Forecast on North-West...")
    output_nw = predict(tensor_nw, model)[0]
    print("Forecast on South-East...")
    output_se = predict(tensor_se, model)[0]

    forecast = np.ones(
        (output_nw.shape[0], RADAR_IMG_SIZE[0], RADAR_IMG_SIZE[1])
    )
    forecast[:, -size_y:, -size_x:] = output_se
    # We assemble the outputs where they overlap enough to avoir disontinuities
    # Hence the 256 offset, to be well inside the receptive field of the model
    forecast[:, : size_y, : size_x - 256] = output_nw[:,:,:-256]
    return forecast


if __name__ == "__main__":

    date = dt.datetime.now(dt.timezone.utc)
    date = date - dt.timedelta(  # round date to 5 minutes
        minutes=date.minute % 5,
        seconds=date.second,
        microseconds=date.microsecond,
    )
    run_date = date - dt.timedelta(minutes=5)  # Remove 5min to be sure data was downloaded

    print(f"---> Making DGMR forecast for date {run_date}")

    file_paths = get_list_files(run_date)
    if not all([f.exists() for f in file_paths]):
        raise FileNotFoundError("Some radar files are not available")

    x_array, mask = get_input_array(file_paths)

    forecast = make_forecast(x_array)

    # Postprocessing : put NaN outside of radar field
    mask = np.tile(mask, (forecast.shape[0], 1, 1))
    forecast = np.where(mask == 1, np.nan, forecast)

    dest_path = PLOT_PATH / run_date.strftime("%Y-%m-%d_%Hh%M.gif")
    plot_gif_forecast(forecast, run_date, dest_path)
