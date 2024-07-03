import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Embedding
from tensorflow.keras.models import load_model

from memory_profiler import profile
gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(gpus[0], True)

# Set a limit on the GPU memory usage (e.g., 4GB)
memory_limit = 3096  # in MB
tf.config.experimental.set_virtual_device_configuration(
gpus[0],
[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

# import resource

# Limit RAM usage to 16GB
ram_limit = 16 * 1024 * 1024 * 1024  # 16GB in bytes
# resource.setrlimit(resource.RLIMIT_AS, (ram_limit, ram_limit))


@profile
def my_function():

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

    # Load required stuff
    #Load pre-trained CNN model
    model = load_model('./output/Model')
    input_shape = model.layers[0].input_shape

    #Define product category
    product_category = ['Debt collection', 'Consumer Loan', 'Mortgage', 'Credit card', 'Credit reporting', 'Student loan', 'Bank account or service', 'Payday loan', 'Money transfers', 'Other financial service', 'Prepaid card']

    #Input consumer complaints data
    complaints = pd.read_csv('./data/new customer complaint.csv')

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
        test = pad_sequences(encoded_test, maxlen=input_shape[1], padding='post')
        test_predict = model.predict(test)
        c.append(product_category[int(np.argmax(test_predict[0]))])


    #Specify output path
    csv_file_path = './output/result.csv'



    # Write the data to the CSV file
    # with open(csv_file_path, 'w', newline='') as csv_file:
    complaints['Predicted Produce'] = c
    complaints.to_csv('./output/result.csv', index=False)

    import matplotlib.pyplot as plt 
    from io import BytesIO
    import seaborn as sns
    sns.set_theme()

    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, BaseDocTemplate, Paragraph, Image, PageBreak, Frame, PageTemplate
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    # start pdf
    doc = SimpleDocTemplate('./output/report2.pdf', pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []

    centered_h3_style = ParagraphStyle(
            name='CenteredH3',
            parent=styles['Heading3'],  # Inherit properties from Heading3 style
            alignment=1,  # 0=left, 1=center, 2=right, 3=justify
        )

    # start report
    elements.append(Paragraph("Report Customer Complaint Classification on Product Category", styles['Title']))
    elements.append(PageBreak())

    sns.countplot(y=c)
    plt.title('Count on Predicted Product Category')
    plt.ylabel('Product Category')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    elements.append(Image(buffer, width=400, height=300))

    doc.build(elements)

if __name__ == '__main__':
    my_function()