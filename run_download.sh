#!/bin/bash

# Activate the micromamba environment
source /usr/local/bin/micromamba
micromamba activate dgmr

python download_data.py