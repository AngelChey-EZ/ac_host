import json
from pathlib import Path
import os
import logging
import pandas as pd
import joblib
from cryptography.fernet import Fernet
import numpy as np
from io import BytesIO
from urllib import request

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
first_input = input_files.pop()

path_input_file = first_input

df = pd.read_csv(path_input_file)
logging.debug('Loaded dataset')

logging.debug('Loading encoders...')
urls = ['https://acentrik-temperory.s3.ap-southeast-1.amazonaws.com/le_country.pkl',
        'https://acentrik-temperory.s3.ap-southeast-1.amazonaws.com/le_bodytype.pkl',
        'https://acentrik-temperory.s3.ap-southeast-1.amazonaws.com/le_brand.pkl'
            ]
les = []

# function to load label encoders
def load_le(url):
    req = request.Request(url)
    response = request.urlopen(req)

    if response.getcode() == 200:
        return joblib.load(BytesIO(response.read()))
    raise Exception(f'Error: Label Encoder could not be loaded from url ({url})')
for url in urls:
    temp = load_le(url)
    les.append(temp)
logging.debug('Encoders loaded')

# function to encrypt the df so that output csv is encrypted
def encrypt_and_save_data(df):
    fernet = Fernet(b'6xDG1u5gSO2lxpRWuBlJuhtB4xGwNyLJFbpI9O-TgC0=')
    encrypted_data = df.map(lambda x: fernet.encrypt(str(x).encode()).decode())
    encrypted_data.to_csv(df_output, index=False)
    return encrypted_data

logging.debug('Begin data proccessing...')
# extract the column
non_date_feature = ['VS: Country/Territory','VS: Bodytype','VS: Sales Brand']
date_feature = [f for f in df.columns if not f.startswith('VS')]
all_feature = non_date_feature + date_feature

# melt features (etc 'Q1 2015') into a value of a colume(quarter_year),
# then the value in the original column 'Q1 2015' becomes the value of another column(sales)
df = pd.melt(df[all_feature], id_vars=non_date_feature, var_name='quarter_year', value_name='Sales')
country = df.iloc[0, 0]

# define output path
model_output = os.path.join(path_output, f'{country}_model.pkl')
df_output = os.path.join(path_output, f'{country}_df.csv')

# function to convert Qx yyyy into date format
logging.debug('Proccssing date...')
def convert_quarter_to_date(quarter_str):
    quarter, year = quarter_str.split()
    year = int(year)
    month = {'Q1': 1, 'Q2': 4, 'Q3': 7, 'Q4': 10}[quarter]
    return pd.Timestamp(year=year, month=month, day=1)

# separate Qx yyyy into integer quarter and year
df['quarter_year'] = df['quarter_year'].apply(convert_quarter_to_date)
df['Year'] = df['quarter_year'].dt.year
df['Quarter'] = df['quarter_year'].dt.quarter
df.drop(columns=['quarter_year'], axis=1, inplace=True)
logging.debug('Proccssing date compelte.')

# change data type for categorical feature
logging.debug('Proccessing categorical features...')
# for col in df.columns[:3]:
#     df[col] = df[col].astype('category')
for x in range(len(les)):
    df[non_date_feature[x]] = les[x].transform(df[non_date_feature[x]])

df = df[non_date_feature + ['Quarter', 'Year', 'Sales']]
logging.debug('Proccssing cetegorical features complete.')

df = df.groupby(df.columns[:-1].tolist(), as_index=False).agg({'Sales': 'sum'})
df = df.sort_values(by=['Year', 'Quarter']).reset_index(drop=True)
df.to_csv('./no.csv', index=False)

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# define the years for training data and the years for testing data
logging.debug('Spliting into training and testing data...')
number_years = df['Year'].nunique()
unique_year = df['Year'].unique()
no_train_years = round(number_years * 0.8)

train_years = unique_year[:no_train_years]
test_years = unique_year[no_train_years:]

x = df.iloc[:, :-1]
y = df['Sales']

# split training and testing data
x_train, x_test = x[x['Year'].isin(train_years)], x[x['Year'].isin(test_years)]
y_train, y_test = y[:x_test.index[0]], y[x_test.index[0]:]
logging.debug('Spliting complete.')

# paramters needed to tune
logging.debug('Optimising model...')
param_grid = {
    'n_estimators': [None, 300, 500, 700],
    'learning_rate': [1.0, 0.8, 0.7],
    'max_depth': [3, 5]
}

xgb = XGBRegressor(random_state=42)

# initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='neg_mean_squared_error', verbose=1)

# fit the grid search
grid_search.fit(x_train, y_train)

# print the best parameters and the corresponding score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)
logging.debug('Optimising complete.')

best_params = grid_search.best_params_
final_model = XGBRegressor(**best_params, random_state=42)
final_model.fit(x_train, y_train)
y_pred = final_model.predict(x_test)
y_pred = np.clip(y_pred, a_min=0, a_max=None)

mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE for {country}: {round(mae, 2)}')
print(f'MAPE for {country}: {round(mape)}%')
print(f'MSE for {country}: {round(mse, 2)}')
print(f'R2 for {country}: {round(r2*100, 2)}%')
print('Final Model:', final_model)

logging.debug('Saving model and encrypted data.')
joblib.dump(final_model, model_output)
encrypt_and_save_data(df)
temp = ['Country', 'Body Type', 'Brand']
for x in range(3):
    df[temp[x]] = les[x].inverse_transform(df.iloc[:, x])
df.to_csv(f'./output/{country}.csv', index=False)
logging.debug(f'Saving complete. Model is saved at {model_output}, encrypted data is saved at {df_output}')

logging.debug("FINISHED ALGORITHM EXECUTION")
