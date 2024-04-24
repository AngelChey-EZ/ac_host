# %% import all library needed

import json
from pathlib import Path
import os
import logging
import pandas as pd
from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.graph_objects as go

# %% Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler()])
logging.info("Starting logging")


# %% Setup Acentrik Standard File Paths

path_input = Path(os.environ.get("INPUTS", "/data/inputs"))
path_output = Path(os.environ.get("OUTPUTS", "/data/outputs"))
path_logs = Path(os.environ.get("LOGS", "/data/logs"))
dids = json.loads(os.environ.get("DIDS", '[]'))
assert dids, f'no DIDS are defined, cannot continue with the algorithm'
did = dids[0]
input_files_path = Path(os.path.join(path_input, did))
input_files = list(input_files_path.iterdir())
first_input = input_files.pop()
path_input_file = first_input
logging.debug(f'got input file: {path_input_file}, {did}, {input_files}')

# Insert your desired Output File Names (can be more than 1 output too with different output file types)
path_output_file = os.path.join(path_output, 'Interactive_Used_Car_Sales(Brand).html')


# %% File Paths Checking
assert path_input_file.exists(), "Can't find required mounted path: {}".format(path_input_file)
assert path_input_file.is_file() | path_input_file.is_symlink(), "{} must be a file.".format(path_input_file)
assert path_output.exists(), "Can't find required mounted path: {}".format(path_output)
logging.debug(f"Selected input file: {path_input_file} {os.path.getsize(path_input_file)/1000/1000} MB")
logging.debug(f"Target output folder: {path_output}")

# %% Load data (can be 1 single file or multiple files)
logging.debug("Loading {}".format(path_input_file))

with open(path_input_file, 'rb') as fh:
    df = pd.read_csv(fh)

logging.debug("Loaded {} records into DataFrame".format(len(df)))

#  %% Insert code logic
df.rename(columns={"name": "brand", 'year': 'year_of_purchase', 'engine': 'engine_displacement_cc', 'mileage': 'mileage_MPG', 'max_power': 'max_power_hp'}, inplace=True)
sold_df = df[df['sold']=='Y']
# %% interactive Brand Sales plot using plotly

fig = make_subplots(rows=1, cols=2, subplot_titles=('Brand Revenue (million): Used Car Sales', 'Brand Sales (Unit): Used Car Sales'))

temp = sold_df.groupby('brand').sum()
temp = temp.sort_values(by='selling_price', ascending=True)
plotA = go.Bar(y=temp.index,
           x=temp['selling_price'],
           name='Sales Revenue',
           orientation='h',)
plotB = go.Bar(y=sold_df['brand'].value_counts(ascending=True).index,
            x=sold_df['brand'].value_counts(ascending=True).values,
            name='Sales Units',
            orientation='h',)
fig.add_trace(
    plotA,
    row=1, col=1
)

fig.add_trace(
    plotB,
    row=1, col=2
)

fig.update_layout(height=800, width=1050, title_text="Brand Sales", yaxis_title='Brand')
fig.update_xaxes(title_text='Revenue (USD million)', row=1, col=1) 
fig.update_xaxes(title_text='Sales (Unit Sold)', row=1, col=2)
pyo.plot(fig, filename=path_output_file, auto_open=False)
logging.debug("Interactive brand sales plot is created and saved.")

# %% Further Logging                  
logging.debug("Built summary of records.")
logging.debug("Interactive brand sales plot saved to {}".format(path_output_file))
logging.debug("FINISHED ALGORITHM EXECUTION")