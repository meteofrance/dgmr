from pathlib import Path

MODEL_PATH = Path("/scratch/shared/dgmr/models")
DATA_PATH = Path("/scratch/shared/beautiful_radar/data/radar/lame_eau_500m_npz")
PLOT_PATH = Path(__file__).parents[1] / "plot"
if not PLOT_PATH.exists():
    PLOT_PATH.mkdir(exist_ok=True)

INPUT_STEPS = 4
PRED_STEPS = 18
TIMESTEP = 5  # minutes
RADAR_IMG_SIZE = (1736, 1736)
INPUT_IMG_SIZE = (1536, 1280)
