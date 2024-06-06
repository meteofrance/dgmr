import datetime as dt
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
dotenv_path = Path(".env")
if dotenv_path.is_file():
    load_dotenv(dotenv_path)

METEO_FRANCE_DATA_PATH = Path(os.getenv("METEO_FRANCE_DATA_PATH", "data"))
METEO_FRANCE_DATA_PATH.mkdir(parents=True, exist_ok=True)

METEO_FRANCE_API_URL = (
    "https://public-api.meteofrance.fr/public/DPRadar/v1/mosaiques/"
    "METROPOLE/observations/LAME_D_EAU/produit?maille=500"
)
METEO_FRANCE_API_KEY = os.getenv("METEO_FRANCE_API_KEY")


max_retries = 20
retry_delay = 3
retry_count = 0
success = False

while retry_count < max_retries:
    response = requests.get(
        METEO_FRANCE_API_URL,
        headers={
            "accept": "application/octet-stream+gzip",
            "apikey": METEO_FRANCE_API_KEY,
        },
    )

    if response.status_code == 200:
        now = dt.datetime.now(dt.timezone.utc)
        now = now - dt.timedelta(  # round date to 5 minutes
            minutes=now.minute % 5,
            seconds=now.second,
            microseconds=now.microsecond,
        )
        output_file = METEO_FRANCE_DATA_PATH / now.strftime("%Y-%m-%d_%Hh%M.h5")
        with open(output_file, "wb") as file:
            file.write(response.content)
        success = True
        break
    else:
        print(f"Attempt {retry_count + 1} failed. Retry in {retry_delay} seconds...")
        retry_count += 1
        time.sleep(retry_delay)

if success:
    print(f"Downloaded successfully {output_file.name}.")
else:
    print(f"Download failed after {max_retries} attempts.")
