import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Embedding
from tensorflow.keras.models import load_model
import csv


#Load pre-trained CNN model
model = load_model('./classifier.model')

#Define product category
product_category = ['Debt collection', 'Consumer Loan', 'Mortgage', 'Credit card', 'Credit reporting', 'Student loan', 'Bank account or service', 'Payday loan', 'Money transfers', 'Other financial service', 'Prepaid card']

#Input consumer complaints data
complaints = pd.read_csv('Consumer Complaint.csv')

#Transform complaints data into list
a = complaints['consumer_complaint_narrative']
b = list(a)

#Transfrom complaints from dataframe column into a long list
list_of_lists = [[x] for x in b]

#Tokenizing text
token = Tokenizer()
token.fit_on_texts(list_of_lists)

#Fit the model to classify all the input complaints data
c = []
for i in list_of_lists:
    token.fit_on_texts(i)
    encoded_test = token.texts_to_sequences(i)
    test = pad_sequences(encoded_test, maxlen=864, padding='post')
    test_predict = model.predict(test)
    c.append(product_category[int(np.argmax(test_predict[0]))])


#Specify output path
csv_file_path = 'output.csv'

# Write the data to the CSV file
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(c)
