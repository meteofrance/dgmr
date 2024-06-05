import datetime as dt
from pathlib import Path
from typing import List

import numpy as np
from scipy.ndimage import zoom
import h5py

from dgmr.settings import DATA_PATH, INPUT_STEPS, TIMESTEP


def get_list_files(date: dt.datetime) -> List[Path]:
    delta = dt.timedelta(minutes=TIMESTEP)
    dates = [date + i * delta for i in range(-INPUT_STEPS + 1, 1)]
    filenames = [d.strftime("%Y_%m_%d_%H_%M.h5") for d in dates]
    return [DATA_PATH / f for f in filenames]


def open_radar_file(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as ds:
        array = np.array(ds["dataset1"]["data1"]["data"])
    return array


def get_input_array(paths: List[Path]) -> np.ndarray:
    arrays = [open_radar_file(path) for path in paths]

    # Put values outside radar field to 0
    mask = np.where(arrays[0] == 65535, 1, 0)
    arrays = [np.where(array == 65535, 0, array) for array in arrays]

    # Rescale to 1km resolution
    arrays = [zoom(array, (0.5, 0.5)) for array in arrays]
    mask = zoom(mask, (0.5, 0.5))

    array = np.stack(arrays)
    array = array / 100 * 12  # Conversion from mm cumulated in 5min to mm/h
    array = np.expand_dims(array, -1)  # Add channel dims
    return array, mask
