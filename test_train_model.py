#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import json
from pathlib import Path
import os
import logging
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Embedding
from sklearn.model_selection import train_test_split
from memory_profiler import profile

@profile
def my_function():
    # Setup logging

    import subprocess
    import time
    import threading

    def log_gpu_memory(interval=10):
        while True:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                                    stdout=subprocess.PIPE)
            usage = result.stdout.decode('utf-8').strip()
            print(f"GPU Memory Usage: {usage}")
            time.sleep(interval)

    # Start logging GPU memory usage in a separate thread
    log_thread = threading.Thread(target=log_gpu_memory, args=(10,), daemon=True)
    log_thread.start()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler()])

    logging.info("Starting logging")


    # Paths

    path_output = Path('./output')
    path_input_file = Path('./data/past_customer_complaints.csv')

    path_output_model = path_output / 'Model'
    path_output_report = os.path.join(path_output, 'report.pdf')

    # Check all paths
    # assert path_input_file.exists(), "Can't find required mounted path: {}".format(path_input_file)
    # assert path_input_file.is_file() | path_input_file.is_symlink(), "{} must be a file.".format(path_input_file)
    # assert path_output.exists(), "Can't find required mounted path: {}".format(path_output)
    # # assert path_logs.exists(), "Can't find required mounted path: {}".format(path_output)
    # logging.debug(f"Selected input file: {path_input_file} {os.path.getsize(path_input_file)/1000/1000} MB")
    logging.debug(f"Target output folder: {path_output}")


    # Load data
    logging.debug("Loading {}".format(path_input_file))

    # with open(path_input_file, 'rb') as fh:
    #     df = pd.read_csv(fh)
    df1 = pd.read_csv('./data/past_complaint_1.csv')
    df2 = pd.read_csv('./data/past_complaint_2.csv')
    df = pd.concat([df1, df2], axis=0)

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
    
    #Saving model
    model.save(path_output_model)

    import matplotlib.pyplot as plt 
    from io import BytesIO
    import seaborn as sns
    sns.set_theme()

    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, BaseDocTemplate, Paragraph, Image, PageBreak, Frame, PageTemplate
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    
    # start pdf
    doc = SimpleDocTemplate(path_output_report, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []

    centered_h3_style = ParagraphStyle(
            name='CenteredH3',
            parent=styles['Heading3'],  # Inherit properties from Heading3 style
            alignment=1,  # 0=left, 1=center, 2=right, 3=justify
        )
    
    # start report
    elements.append(Paragraph("Report Model Training: Finance Consumer Complaints Classification", styles['Title']))
    elements.append(PageBreak())
    
    # plot graphs
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Determine the number of epochs
    epochs = range(1, len(loss) + 1)

    # Plotting the training and validation loss
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    elements.append(Image(buffer, width=400, height=300))

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Determine the number of epochs
    epochs = range(1, len(loss) + 1)

    # Plotting the training and validation loss
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    elements.append(Image(buffer, width=400, height=300))
    
    # build the document
    doc.build(elements)


    logging.debug("Built summary of records.")

    

    logging.debug("Wrote results to {}".format(path_output_model))

    logging.debug("FINISHED ALGORITHM EXECUTION")

if __name__ == '__main__':
    my_function()