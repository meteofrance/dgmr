import ssl
from datetime import datetime
from pathlib import Path

import cartopy.feature as cfeature
import gif
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from cartopy.crs import PlateCarree, Stereographic
from tqdm import trange

from dgmr.settings import INPUT_STEPS, PRED_STEPS, TIMESTEP

ssl._create_default_https_context = ssl._create_unverified_context


def hex_to_rgb(hex):
    """Converts a hexadecimal color to RGB."""
    return tuple(int(hex[i : i + 2], 16) / 255 for i in (0, 2, 4))


COLORS_REFLECTIVITY = [  # 14 colors
    hex_to_rgb("E5E5E5"),
    hex_to_rgb("6600CBFF"),
    hex_to_rgb("0000FFFF"),
    hex_to_rgb("00B2FFFF"),
    hex_to_rgb("00FFFFFF"),
    hex_to_rgb("0EDCD2FF"),
    hex_to_rgb("1CB8A5FF"),
    hex_to_rgb("6BA530FF"),
    hex_to_rgb("FFFF00FF"),
    hex_to_rgb("FFD800FF"),
    hex_to_rgb("FFA500FF"),
    hex_to_rgb("FF0000FF"),
    hex_to_rgb("991407FF"),
    hex_to_rgb("FF00FFFF"),
]
"""list of str: list of colors for the orange-blue cumulated rainfall colormap"""

CMAP = mcolors.ListedColormap(COLORS_REFLECTIVITY)
"""ListedColormap : reflectivity colormap from synopsis"""

BOUNDARIES = [
    0,
    0.1,
    0.4,
    0.6,
    1.2,
    2.1,
    3.6,
    6.5,
    12,
    21,
    36,
    65,
    120,
    205,
    360,
]
"""list of float: boundaries of the reflectivity colormap"""

NORM = mcolors.BoundaryNorm(BOUNDARIES, CMAP.N, clip=True)
"""BoundaryNorm: norm for the reflectivity colormap"""

DOMAIN = {
    "upper_left": (-9.965, 53.670),
    "lower_right": (11.976055, 37.457460),
    "upper_right": (17.564203, 52.548138),
    "lower_left": (-6.715173, 38.144933),
}


def domain_to_extent(domain):
    crs = Stereographic(central_latitude=45)
    lower_right = crs.transform_point(*domain["lower_right"], PlateCarree())
    upper_right = crs.transform_point(*domain["upper_right"], PlateCarree())
    lower_left = crs.transform_point(*domain["lower_left"], PlateCarree())
    maxy, miny = upper_right[1], lower_left[1]
    minx, maxx = lower_left[0], lower_right[0]
    return (minx, maxx, miny, maxy)


EXTENT = domain_to_extent(DOMAIN)


@gif.frame
def plot_forecast(y_hat: list, run_date: datetime, delta: int):
    """Plots one frame of the forecast gif.
    y_hat: np.ndarray = prediction made by the model
    date: datetime
    """
    fig = plt.figure(figsize=(12, 12), dpi=300)
    ax = plt.axes(projection=Stereographic(central_latitude=45))

    plot_kwargs = {
        "extent": EXTENT,
        "interpolation": "none",
        "norm": NORM,
        "cmap": CMAP,
    }

    # Prediction
    img = ax.imshow(y_hat, **plot_kwargs)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black")
    ax.coastlines(resolution="50m", color="black", linewidth=1)
    ax.set_title("Forecast", fontsize=20)

    # Colorbar
    cb = fig.colorbar(img, ax=ax, orientation="horizontal", fraction=0.04, pad=0.05)
    cb.set_label(label="Precipitations (mm/h)", fontsize=15)

    run_date = run_date.strftime("%Y-%m-%d %H:%M")
    fig.suptitle(f"Run: {run_date} | + {delta:02} min", fontsize=20, y=0.97)


def plot_gif_forecast(y_hat: np.ndarray, date: datetime, save_path: Path):
    """Plots a gif of the forecast."""
    images = []
    for i in trange(PRED_STEPS + INPUT_STEPS):
        delta = (i - INPUT_STEPS + 1) * TIMESTEP
        images.append(plot_forecast(y_hat[i], date, delta))
    gif.save(images, str(save_path), duration=200)
