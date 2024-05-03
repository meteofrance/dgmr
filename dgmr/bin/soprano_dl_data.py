"""Télécharge une image radar depuis la BDI et l'envoie sur priam-sidev."""

import getpass
import netrc
import subprocess
from datetime import datetime, timedelta
from ftplib import FTP
from pathlib import Path

import numpy as np
from osgeo import gdal

DEST_SERVER = "priam-sidev.meteo.fr"
DEST_FOLDER = "/scratch/shared/RADAR_DATA/lame_eau_npz/Hexagone/"

folder = Path().absolute()
date = datetime.now()
date = date - timedelta(  # round date to 5 minutes
    minutes=date.minute % 5, seconds=date.second, microseconds=date.microsecond
)
date = date - timedelta(minutes=10)  # remove 10 minutes, to be sure image is available
date_str = date.strftime("%Y%m%d%H%M")
print(f"---> Download radar image from BDI for date {date}")

# Write BDI request
request = f"type_produit_id:RD_CPO_NAT100\n\
type_production_id:CUMUL5\n\
producteur_gene_id:SYCOMORE\n\
nom_vue_id:V_EUR_COMPO\n\
param_id:LAME_DEAU\n\
format_id:BUFR_F\n\
gzip:N\n\
info:O\n\
min_dat_reseau:{date.strftime('%Y%m%d%H%M%S')}\n\
max_dat_reseau:{date.strftime('%Y%m%d%H%M%S')}\n\
min_dat_validite:{date.strftime('%Y%m%d%H%M%S')}\n\
max_dat_validite:{date.strftime('%Y%m%d%H%M%S')}\n\
plus_valide:N\n\
bstr:O"
file_request = Path("bdi_request.txt")
with open(file_request, "w") as file:
    file.write(request)

# Download from BDI
subprocess_kwargs = {
    "shell": True,
    "stdout": subprocess.PIPE,
    "stderr": subprocess.PIPE,
}
process_bdi = subprocess.Popen(  # nosec B602
    f"RdImage -f{file_request}", **subprocess_kwargs  # nosec B602
)  # nosec B602
process_bdi.wait(timeout=60)
bufr_file = list(folder.glob(f"RD_CPO*{date_str}*"))[0]
print("Bufr file : ", bufr_file.name)

# Convert to geotiff
gtiff_file = folder / f"{date_str}.gtiff"
process_bufr2geotiff = subprocess.Popen(  # nosec B602
    f"bufrtogeotiff {bufr_file} {gtiff_file}", shell=True  # nosec B602
)  # nosec B602
process_bufr2geotiff.wait(timeout=60)

# Convert to npz
npz_file = folder / f"{date_str}.npz"
gdal_data = gdal.Open(str(gtiff_file))
array = gdal_data.GetRasterBand(3).ReadAsArray()
np.savez_compressed(npz_file, array)
print("File converted to NPZ : ", npz_file.name)

# FTP upload to server
with FTP(DEST_SERVER) as ftp:
    netrc_inst = netrc.netrc(f"/home/labia/{getpass.getuser()}/.netrc")
    creds = netrc_inst.authenticators(DEST_SERVER)
    ftp.login(user=creds[0], passwd=creds[2])
    ftp.cwd(DEST_FOLDER)
    with open(npz_file, "rb") as file:
        ftp.storbinary(f"STOR {npz_file.name}", file)
print(f"File sent to {DEST_SERVER}")

# Remove everything
for f in list(folder.glob("RD_CPO*")):
    f.unlink()
gtiff_file.unlink()
npz_file.unlink()
file_request.unlink()

print("Done !")
