import ssl
from datetime import datetime

import cartopy.feature as cfeature
import gif
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from cartopy.crs import PlateCarree, Stereographic
from tqdm import trange

from dgmr.settings import PLOT_PATH, PRED_STEPS, TIMESTEP

ssl._create_default_https_context = ssl._create_unverified_context


def hex_to_rgb(hex):
    """Converts a hexadecimal color to RGB."""
    return tuple(int(hex[i : i + 2], 16) / 255 for i in (0, 2, 4))


COLORS_REFLECTIVITY = [  # 14 colors
    hex_to_rgb("FFFFFFFF"),
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
    "lower_right": (10.259217, 39.46785),
    "upper_right": (14.564706, 53.071644),
    "lower_left": (-6.977881, 39.852361),
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


def plot_error_per_leadtime(y_hat: list, y: np.ndarray, run_date: datetime):
    """
    Plot side by side the prediction, the ground truth and the difference between them.
    """
    leadtimes_to_plot = (
        2,
        5,
        11,
    )  # , 17)  # Corresponding to +15, +30, +60 and +90 minutes

    subplot_kw = {"projection": Stereographic(central_latitude=45)}
    fig, axs = plt.subplots(
        nrows=len(leadtimes_to_plot),
        ncols=3,
        figsize=(10, 10),
        subplot_kw=subplot_kw,
        dpi=300,
    )
    plot_kwargs = {"extent": EXTENT, "interpolation": "none"}
    axs[0, 0].set_title("Observation", fontsize=20)
    axs[0, 1].set_title("Prévision", fontsize=20)
    axs[0, 2].set_title("Observation - Prévision", fontsize=20)

    for i, leadtime in enumerate(leadtimes_to_plot):
        # Observation
        y_leadtime = y[leadtime]
        img_y = axs[i, 0].imshow(y_leadtime, norm=NORM, cmap=CMAP, **plot_kwargs)
        axs[i, 0].add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black")
        axs[i, 0].coastlines(resolution="50m", color="black", linewidth=1)
        axs[i, 0].set_ylabel(f"+ {leadtime * TIMESTEP} min")

        # Prediction
        y_hat_leadtime = y_hat[leadtime]
        axs[i, 1].imshow(y_hat_leadtime, norm=NORM, cmap=CMAP, **plot_kwargs)
        axs[i, 1].add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black")
        axs[i, 1].coastlines(resolution="50m", color="black", linewidth=1)

        # Difference
        error_leadtime = y_leadtime - y_hat_leadtime
        img_error = axs[i, 2].imshow(
            error_leadtime, cmap="PiYG", vmin=-20, vmax=20, **plot_kwargs
        )
        axs[i, 2].add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black")
        axs[i, 2].coastlines(resolution="50m", color="black", linewidth=1)

    fig.suptitle(f"Run: {run_date.strftime('%Y-%m-%d %H:%M')}", fontsize=25, y=0.95)
    plt.subplots_adjust(wspace=0.1, hspace=0.05, bottom=0.1)

    # Colorbar Obs + Prev
    cb = fig.colorbar(
        img_y, ax=axs[-1, :2], orientation="horizontal", fraction=0.04, pad=0.05
    )
    cb.set_label(label="Lame d'eau (mm/h)", fontsize=15)

    # Colorbar Erreur
    cb_diff = fig.colorbar(
        img_error, ax=axs[-1, 2], orientation="horizontal", fraction=0.04, pad=0.05
    )
    cb_diff.set_label(label="Différence (mm/h)", fontsize=15)

    dest_path = PLOT_PATH / "last_error.png"
    plt.savefig(dest_path)


@gif.frame
def plot_predict(y_hat: list, y: np.ndarray, run_date: datetime, delta: int):
    """
    y_hat: np.ndarray = prediction made by the model
    y: np.ndarray = ground truth
    extent: tuple = tuple containing the coordinates of the crop.
    date: datetime
    """
    subplot_kw = {"projection": Stereographic(central_latitude=45)}
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(15, 7.5),
        subplot_kw=subplot_kw,
        dpi=300,
    )
    plot_kwargs = {
        "extent": EXTENT,
        "interpolation": "none",
        "norm": NORM,
        "cmap": CMAP,
    }

    # Observation
    img = axs[0].imshow(y, **plot_kwargs)
    axs[0].add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black")
    axs[0].coastlines(resolution="50m", color="black", linewidth=1)
    axs[0].set_title("Observations", fontsize=20)

    # Prediction
    img = axs[1].imshow(y_hat, **plot_kwargs)
    axs[1].add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black")
    axs[1].coastlines(resolution="50m", color="black", linewidth=1)
    axs[1].set_title("Prévisions", fontsize=20)

    # Colorbar
    cb = fig.colorbar(
        img, ax=axs[:2], orientation="horizontal", fraction=0.04, pad=0.05
    )
    cb.set_label(label="Précipitations en mm/h", fontsize=15)

    run_date = run_date.strftime("%Y-%m-%d %H:%M")
    fig.suptitle(f"Run: {run_date} | + {delta:02} min", fontsize=20, y=0.97)


def plot_gif(y_hat: np.ndarray, y: np.ndarray, date: datetime):
    """Plot gif of prediction versus ground truth."""
    images = []
    for i in trange(1, PRED_STEPS + 1):
        images.append(plot_predict(y_hat[i - 1], y[i - 1], date, i * TIMESTEP))
    dest_path = PLOT_PATH / "last_gif.gif"
    gif.save(images, dest_path, duration=200)
