import json
from pathlib import Path
import os
import logging
import pandas as pd


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler()])
logging.info("Starting logging")

# define paths
path_input = Path(os.environ.get("INPUTS", "/data/inputs"))
path_output = Path(os.environ.get("OUTPUTS", "/data/outputs"))
path_logs = Path(os.environ.get("LOGS", "/data/logs"))
dids = json.loads(os.environ.get("DIDS", '[]'))
did = dids[0]
input_files_path = Path(os.path.join(path_input, did))
input_files = list(input_files_path.iterdir())
print('print(input files)')
print(input_files)

first_input = input_files.pop()



path_input_file = first_input
file_output = os.path.join(path_output, f'df.csv')

df = pd.read_csv(path_input_file)

# Get all environment variables
env_vars = os.environ
print('list env')
# Print each environment variable
for key, value in env_vars.items():
    print(f'{key}: {value}')

df.to_csv(file_output, index=False)