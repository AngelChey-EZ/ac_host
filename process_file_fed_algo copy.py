import json
from pathlib import Path
import os
import logging
import pandas as pd
import joblib

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler()])
logging.info("Starting logging")

# define paths
path_input = Path(os.environ.get("INPUTS", "/data/inputs"))
path_output = Path('./output')
path_logs = Path(os.environ.get("LOGS", "/data/logs"))
dids = json.loads(os.environ.get("DIDS", '[]'))
did = dids[0]
input_files_path = Path(os.path.join(path_input, did))
input_files = list(input_files_path.iterdir())
first_input = input_files.pop()

path_input_file = first_input

scaler_output = os.path.join(path_output, 'Scaler.pkl')
encoder_output = os.path.join(path_output, 'Encoder.pkl')
model_output = os.path.join(path_output, 'model.joblib')
mental_health_output = os.path.join(path_output, 'mental_health.csv')
crime_output = os.path.join(path_output, 'crime.csv')

print(path_input_file)
files = [f for f in path_input_file.iterdir() if f.is_file()]
print(files)
print(path_input_file.is_file(), path_input_file.is_dir())
with open(path_input_file, 'rb') as fh:
        df = pd.read_csv(fh)
result = df.to_csv(mental_health_output, index=False)

logging.debug(f'Loaded file {path_input_file}.')

# check file extension and if its pkl or joblib, load and dump it, if its csv save it
# if path_input_file.lower().endswith('.pkl'):
#     if 'scaler' in path_input_file.lower():
#         scaler = joblib.load(path_input_file)
#         result = joblib.dump(scaler, scaler_output)
#         logging.debug(f'Saved scaler at {result}.')
        
#     elif 'encoder' in path_input_file.lower():
#         encoder = joblib.load(path_input_file)
#         result = joblib.dump(encoder, encoder_output)
#         logging.debug(f'Saved encoder at {result}.')
        
#     else:
#         raise Exception(f'File input "{path_input_file}" is not excepted, please ensure that the file input is correct!')
    
# elif path_input_file.lower().endswith('.joblib') and 'model' in path_input_file.lower():
#         model = joblib.load(path_input_file)
#         result = joblib.dump(model, model_output)
#         logging.debug(f'Saved model at {result}.')

# elif path_input_file.lower().endswith('.csv'):
#     with open(path_input_file, 'rb') as fh:
#         df = pd.read_csv(fh)
#     logging.debug('CSV file detected.')

# else:
#     raise Exception(f'File input "{path_input_file}" is not excepted, please ensure that the file input is correct!')

# # process the dataframe, if needed, if a csv is read and saved in variable 'df'
# if 'df' in locals():
#     if 'criminal_records' in df.columns:
#         logging.debug('Processing criminal records...')
#         # filter records with criminal records
#         df = df[~df['criminal_records'].isna()].rename(columns={'criminal_records': 'crime'})
#         df = df[['id', 'crime']]
#         df.to_csv(crime_output, index=False) 
#         logging.debug(f'Processed criminal records saved at {crime_output}.')
    
#     elif 'category' in df.columns and 'illness' in df.columns:
#         logging.debug('Processing mental health records...')
#         # filter records with mental illness
#         df = df[df['category'] == 'Psychiatry'].rename(columns={'category': 'mental_health'})
#         df = df[['id', 'mental_health']]
#         result = df.to_csv(mental_health_output, index=False) 
#         logging.debug(f'Processed mental health records saved at {mental_health_output}.')
    
#     elif 'blacklist' in path_input_file.lower():
#         logging.debug('Processing blacklist records...')
#         filename = Path(path_input_file).stem
#         blacklist_output = os.path.join(path_output, f'{filename}.csv')
#         result = df.to_csv(blacklist_output, index=False) 
#         logging.debug(f'Processed blacklist records saved at {blacklist_output}.')
    
#     elif 'registration' in path_input_file.lower():
#         logging.debug('Processing registration records...')
#         filename = Path(path_input_file).stem
#         registration_output = os.path.join(path_output, f'{filename}.csv')
#         result = df.to_csv(registration_output, index=False)
#         logging.debug(f'Processed registration records saved at {registration_output}.')
        
#     else:
#         raise Exception(f'File input "{path_input_file}" is not excepted, please ensure that the file input is correct!')
    
#     del df
logging.debug("FINISHED ALGORITHM EXECUTION")