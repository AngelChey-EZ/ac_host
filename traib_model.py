#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os
import logging
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Embedding
from sklearn.model_selection import train_test_split


# %% Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler()])

logging.info("Starting logging")


# %% Paths
path_input = Path(os.environ.get("INPUTS", "/data/inputs"))
path_output = Path(os.environ.get("OUTPUTS", "/data/outputs"))
path_logs = Path(os.environ.get("LOGS", "/data/logs"))
dids = json.loads(os.environ.get("DIDS", '[]'))
assert dids, f'no DIDS are defined, cannot continue with the algorithm'

did = dids[0]
input_files_path = Path(os.path.join(path_input, did))
input_files = list(input_files_path.iterdir())
first_input = input_files.pop()
# assert len(input_files) == 1, "Currently, only 1 input file is supported."
path_input_file = first_input
logging.debug(f'got input file: {path_input_file}, {did}, {input_files}')
path_output_file = path_output / 'Model'

# %% Check all paths
assert path_input_file.exists(), "Can't find required mounted path: {}".format(path_input_file)
assert path_input_file.is_file() | path_input_file.is_symlink(), "{} must be a file.".format(path_input_file)
assert path_output.exists(), "Can't find required mounted path: {}".format(path_output)
# assert path_logs.exists(), "Can't find required mounted path: {}".format(path_output)
logging.debug(f"Selected input file: {path_input_file} {os.path.getsize(path_input_file)/1000/1000} MB")
logging.debug(f"Target output folder: {path_output}")


# %% Load data
logging.debug("Loading {}".format(path_input_file))

with open(path_input_file, 'rb') as fh:
    df = pd.read_csv(fh)

logging.debug("Loaded {} records into DataFrame".format(len(df)))


#Removing unwanted data
data = df[['product', 'consumer_complaint_narrative']]

#Removing null values
data.dropna(inplace=True)

#Changing dataframe column to list
text = data['consumer_complaint_narrative'].tolist()

#Encoding labels for product category
product_category = list(data['product'].unique())
labels = []
for i in data['product']:
    labels.append(product_category.index(i))

#Tokenizing text
token = Tokenizer()
token.fit_on_texts(text)
vocabs = token.index_word
vocabs_len = len(token.word_index) + 1

#Mapping token back to text
encoded_text = token.texts_to_sequences(text)

#Preparing data for modeling
max_length = max(len(x) for x in encoded_text)
X = pad_sequences(encoded_text, maxlen=max_length, padding='post')
X_train, X_test, y_train, y_test = train_test_split(X, np.array(labels), test_size=0.2, random_state=42)


#Building and training model
vec_size = 300

model = Sequential()
model.add(Embedding(vocabs_len, vec_size, input_length=max_length))

model.add(Conv1D(kernel_size=64, filters=8, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(16, activation='relu'))

model.add(GlobalMaxPooling1D())

model.add(Dense(len(product_category), activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs = 5, validation_data=(X_test, y_test))



logging.debug("Built summary of records.")

#Saving model
model.save(path_output_file)

logging.debug("Wrote results to {}".format(path_output_file))

logging.debug("FINISHED ALGORITHM EXECUTION")
