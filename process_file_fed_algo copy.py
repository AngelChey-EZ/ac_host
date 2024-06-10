import json
from pathlib import Path
import os
import logging
import pandas as pd
import joblib
import sklearn

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
registration_output = os.path.join(path_output, 'registration.csv')


try:
    # try to read as csv
    with open(path_input_file, 'rb') as fh:
        df = pd.read_csv(fh)
except Exception:
    # try to use joblib to load
    try:
        with open(path_input_file, 'rb') as fh:
            others = joblib.load(fh)

    except Exception:
        raise Exception(f'File input "{path_input_file}" is not excepted, please ensure that the file input is correct!')

logging.debug(f'Loaded file {path_input_file}.')

registration_columns = ['id', 'name', 'gender', 'birth_year', 
                        'nationality', 'job_sector', 'income', 
                        'smoke', 'alcohol', 'drug', 'debt', 
                        'education', 'relationship', 'household_composition', 
                        'housing_tenure']

# list of actions and checks if its a csv file
if 'df' in locals():
    if 'criminal_records' in df.columns:
        logging.debug('Processing criminal records...')
        # filter records with criminal records
        df = df[~df['criminal_records'].isna()].rename(columns={'criminal_records': 'crime'})
        df = df[['id', 'crime']]
        df.to_csv(crime_output, index=False) 
        logging.debug(f'Processed criminal records saved at {crime_output}.')
    
    elif 'category' in df.columns and 'illness' in df.columns:
        logging.debug('Processing mental health records...')
        # filter records with mental illness
        df = df[df['category'] == 'Psychiatry'].rename(columns={'category': 'mental_health'})
        df = df[['id', 'mental_health']]
        result = df.to_csv(mental_health_output, index=False) 
        logging.debug(f'Processed mental health records saved at {mental_health_output}.')
    
    elif df.columns.tolist() == ['id', 'name', 'gender', 'birth_year', 'nationality']:
        # check if its blacklist
        logging.debug('Processing blacklist records...')
        filename = 'blacklist_' + path_input_file.stem[-6:-2]
        blacklist_output = os.path.join(path_output, f'{filename}.csv')
        result = df.to_csv(blacklist_output, index=False) 
        logging.debug(f'Processed blacklist records saved at {blacklist_output}.')
    
    elif df.columns.tolist() == registration_columns:
        # check if its registration data
        logging.debug('Processing registration records...')
        result = df.to_csv(registration_output, index=False)
        logging.debug(f'Processed registration records saved at {registration_output}.')

    else:
        raise Exception(f'File input "{path_input_file}" is not excepted, please ensure that the file input is correct!')

# list of checks and actions of its not csv file
if 'others' in locals():
    if type(others) == sklearn.preprocessing._data.MinMaxScaler:
        scaler = joblib.load(path_input_file)
        result = joblib.dump(scaler, scaler_output)
        logging.debug(f'Saved scaler at {result}.')
    
    elif type(others) == sklearn.preprocessing._encoders.OneHotEncoder:
        encoder = joblib.load(path_input_file)
        result = joblib.dump(encoder, encoder_output)
        logging.debug(f'Saved encoder at {result}.')
    
    elif type(others) == sklearn.ensemble._voting.VotingClassifier:
        model = joblib.load(path_input_file)
        result = joblib.dump(model, model_output)
        logging.debug(f'Saved model at {result}.')

    else:
        raise Exception(f'File input "{path_input_file}" is not excepted, please ensure that the file input is correct!')
    
logging.debug("FINISHED ALGORITHM EXECUTION")
