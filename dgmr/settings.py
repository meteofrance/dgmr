from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("METEO_FRANCE_DATA_PATH")
DATA_PATH = Path("data/") if DATA_PATH == "" else Path(DATA_PATH)

MODEL_PATH = Path(os.getenv("DGMR_MODEL_PATH"))

PLOT_PATH = Path("DGMR_PLOT_PATH")
if not PLOT_PATH.exists():
    PLOT_PATH.mkdir(exist_ok=True)

INPUT_STEPS = 4
PRED_STEPS = 18
TIMESTEP = 5  # minutes
RADAR_IMG_SIZE = (1736, 1736)
INPUT_IMG_SIZE = (1536, 1280)
