import pandas as pd
import os
import joblib
import numpy as np
from io import BytesIO
from urllib import request
import logging
from pathlib import Path
from urllib.parse import unquote
import json

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler()])
logging.info("Starting logging")

# LOAD FILE ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
path_input = Path(
    os.path.join(os.environ.get("INPUTS", "/data/inputs"), "algoCustomData.json")
)
path_output = Path(os.environ.get("OUTPUTS", "/data/outputs"))
path_logs = Path(os.environ.get("LOGS", "/data/logs"))

path_output_file = os.path.join(path_output, 'automotive_predict_sales.csv')
report_output_path = os.path.join(path_output, 'report.pdf')
model_output_path = os.path.join(path_output, 'stacking_regression.pkl')


algoCustomData = {}

logging.debug("Loading input files...")
with open(path_input, "r") as json_file:
    algoCustomData = json.load(json_file)

print(algoCustomData)
path_input = Path(
    os.path.join(os.environ.get("INPUTS", "/data/inputs"))
)

input_files = list(path_input.iterdir())

# Get all environment variables
env_vars = os.environ

# Print each environment variable
for key, value in env_vars.items():
    print(f'{key}: {value}')