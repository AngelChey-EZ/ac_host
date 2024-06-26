import pandas as pd
import os
import joblib
from cryptography.fernet import Fernet
import numpy as np
from io import BytesIO
from urllib import request
import logging
from pathlib import Path
from urllib.parse import unquote
from io import StringIO
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


algoCustomData = {}

logging.debug("Loading input files...")
with open(path_input, "r") as json_file:
    algoCustomData = json.load(json_file)

result_data = algoCustomData["resultUrls"]

dfs = []
models = []

# function to decrypt data
def load_and_decrypt_df(data):
    df = pd.read_csv(data)
    fernet = Fernet(b'6xDG1u5gSO2lxpRWuBlJuhtB4xGwNyLJFbpI9O-TgC0=')
    decrypted_data = df.map(lambda x: fernet.decrypt(x.encode()).decode())
    return decrypted_data

for job_data in result_data:
    try:
        url = job_data["job_url"]
        headers = job_data["job_headers"]
        req = request.Request(url, headers=headers)  # Create a request with headers
        response = request.urlopen(req)
        
        if response.getcode() == 200:
        
            # retrive the file name
            content_disposition = response.headers['content-disposition']
            filename_index = content_disposition.find('filename=')
            if filename_index != -1:
                filename = content_disposition[filename_index+len('filename='):]
                filename = unquote(filename)  
                filename = filename.strip('"') 
            
            print(filename)
                
            # load model and transformer
            if filename.lower().endswith('.pkl'):
                country_name, _ = filename.split('_')
                model = joblib.load(BytesIO(response.read()))
                print(model)
                models.append((country_name, model))

            elif filename.lower().endswith('.csv'):
                csv_data = response.read().decode("utf-8")
                temp = load_and_decrypt_df(StringIO(csv_data))
                dfs.append(temp)
        

    except Exception as e:
        raise Exception(f"Error fetching data from URL: {url}, error: {e}")
logging.debug('Loaded all input files.')
print(models)
print(dfs)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

logging.info("Loading encoders...")
# urls for label encoder
urls = ['https://acentrik-temperory.s3.ap-southeast-1.amazonaws.com/le_country.pkl',
        'https://acentrik-temperory.s3.ap-southeast-1.amazonaws.com/le_bodytype.pkl',
        'https://acentrik-temperory.s3.ap-southeast-1.amazonaws.com/le_brand.pkl'
            ]
les = []

# load label encoder
def load_le(url):
    req = request.Request(url)
    response = request.urlopen(req)

    if response.getcode() == 200:
        return joblib.load(BytesIO(response.read()))
    raise Exception(f'Error: Label Encoder could not be loaded from url ({url})')
for url in urls:
    temp = load_le(url)
    les.append(temp)
logging.info("Loaded encoders")

logging.info("Pocessing data...")
# combine data
df = pd.concat(dfs, axis=0)
df = df.sort_values(by=['Year', 'Quarter']).reset_index(drop=True)
df = df.astype('int')

# split the data into train and test set
number_years = df['Year'].nunique()
unique_year = df['Year'].unique()
no_train_years = round(number_years * 0.8)

train_years = unique_year[:no_train_years]
test_years = unique_year[no_train_years:]

x = df.iloc[:, :-1]
y = df['Sales']

print(x)

x_train, x_test = x[x['Year'].isin(train_years)], x[x['Year'].isin(test_years)]
y_train, y_test = y[:x_test.index[0]], y[x_test.index[0]:]
logging.info("Pocessed data")

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

logging.info("Training stacking regressor...")
# train stacking regressor
stacking_regressor = StackingRegressor(
    estimators=models,
    final_estimator=LinearRegression(),
    passthrough=True,
    verbose=1
)
stacking_regressor.fit(x_train, y_train)

# evaluate model
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
y_pred = stacking_regressor.predict(x_test)
y_pred = np.clip(y_pred, a_min=0, a_max=None)

mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE : {round(mae, 2)}')
print(f'MAPE : {round(mape)}%')
print(f'MSE : {round(mse, 2)}')
print(f'R2 : {round(r2*100, 2)}%')
print('Final Model:', stacking_regressor)
logging.info("Trained stacking regressor")

logging.info("Predicting future sales...")
# create prediction data, for the next 4 quarters
countries = df['VS: Country/Territory'].unique()
predict_df = []

for c in countries:
    temp = df[df['VS: Country/Territory']==c]

    for bt in temp['VS: Bodytype'].unique():
        temp2 = temp[temp['VS: Bodytype']==bt]

        for b in temp2['VS: Sales Brand'].unique():
            temp3 = temp2[temp2['VS: Bodytype']==bt]

            last_q = temp3['Quarter'].values[-1]
            last_y = temp3['Year'].values[-1]

            # for each country, body type brand combi, create prediction for next 4 quarters
            for _ in range(4):
                if last_q == 4:
                    last_q = 1
                    last_y += 1
                else:
                    last_q += 1

                # country, body type, brand, quarter, year
                predict_df.append([c, bt, b, last_q, last_y])
                
predict_df = pd.DataFrame(predict_df, 
                          columns=['VS: Country/Territory','VS: Bodytype','VS: Sales Brand','Quarter','Year'])

# process predicted values
predicted_sales = stacking_regressor.predict(predict_df)
predicted_sales = np.clip(predicted_sales, a_min=0, a_max=None)
predict_df['Predicted Sales'] = predicted_sales
predict_df['Predicted Sales'] = predict_df['Predicted Sales'].astype('int')

# transform encoded value back to original value
temp = ['Country', 'Body Type', 'Brand']
for x in range(3):
    predict_df[temp[x]] = les[x].inverse_transform(predict_df.iloc[:, x])
logging.info("Predicted future sales")

logging.info("Saving Result...")
output_df = predict_df[['Country', 'Body Type', 'Brand', 'Quarter', 'Year', 'Predicted Sales']]
output_df.to_csv(path_output_file, index=False)
logging.debug(f'Result is saved to {path_output_file}')

######################## GENERATE REPORT ########################
logging.info("Generating Report...")
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import seaborn as sns
sns.set_theme()

from reportlab.lib.pagesizes import A4
from reportlab.platypus import BaseDocTemplate, SimpleDocTemplate, Paragraph, Table, TableStyle, Image, PageBreak, Frame, PageTemplate, FrameBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

########################################################################
temp = predict_df.groupby(['Country'], as_index=False).agg({'Predicted Sales': 'sum'}).sort_values(by='Predicted Sales', ascending=False)
country_ranking = temp['Country']


########################################################################
# formatting functions
import re

def add_commas(number):
    # Convert the number to a string if it isn't already
    num_str = str(number)
    # Use regular expression to insert commas every three digits from the back
    num_with_commas = re.sub(r'(?<=\d)(?=(\d{3})+(?!\d))', ',', num_str)
    return num_with_commas

def format_yaxis(value, tick_number):
    # Custom formatter function to add units based on the value
    if value >= 1_000_000:
        return f'{value / 1_000_000:.1f}M'
    elif value >= 1_000:
        return f'{value / 1_000:.1f}K'
    else:
        return str(value)
########################################################################

# start pdf
doc = BaseDocTemplate(report_output_path, pagesize=A4)
styles = getSampleStyleSheet()

frame_full = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='full')
frame_left = Frame(doc.leftMargin, doc.bottomMargin, (doc.width / 2) - 6, doc.height, id='left_col')
frame_right = Frame(doc.leftMargin + (doc.width / 2) + 6, doc.bottomMargin, (doc.width / 2) - 6, doc.height, id='right_col')

# Create page templates
single_col_template = PageTemplate(id='OneCol', frames=[frame_full])
two_col_template = PageTemplate(id='TwoCol', frames=[frame_left, frame_right])
doc.addPageTemplates([single_col_template, two_col_template])

elements = []

centered_h3_style = ParagraphStyle(
        name='CenteredH3',
        parent=styles['Heading3'],  # Inherit properties from Heading3 style
        alignment=1,  # 0=left, 1=center, 2=right, 3=justify
    )

# start report
elements.append(Paragraph("Report on Automotive Sales Prediction", styles['Title']))
elements.append(Paragraph('Note: All sales refer to sales volume!', styles['Italic']))
elements.append(PageBreak())


# top 20 sales Globally
year_predicted_sales = predict_df.groupby(['Country', 'Body Type', 'Brand', 'Year'], as_index=False).agg({'Predicted Sales': 'sum'})
year_predicted_sales = year_predicted_sales.sort_values(by='Predicted Sales', ascending=False)
year_predicted_sales['Predicted Sales'] = year_predicted_sales['Predicted Sales'].apply(add_commas)
data_list = [year_predicted_sales.columns.values.tolist()] + year_predicted_sales.head(20).values.tolist()
table = Table(data_list)

style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 1), (-1, -1), 12),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ('TOPPADDING', (0, 1), (-1, -1), 6),
])

table.setStyle(style)
elements.append(Paragraph(f"This report include sales prediction for {len(year_predicted_sales['Country'].unique())} countries. <br />", styles['Heading3']))
elements.append(Paragraph("Top 20 Predicted Sales Globally", centered_h3_style))
elements.append(table)
elements.append(PageBreak())

# line graphs for each country
def country_line_plot():
    temp1 = predict_df.groupby(['Country'], as_index=False).agg({'Predicted Sales': 'sum'}).sort_values(by='Predicted Sales', ascending=False)
    country_ranking = temp1['Country']
    temp_df = []
    for x in country_ranking:
        temp = predict_df[predict_df['Country'] == x]
        temp = temp.groupby(['Quarter', 'Year']).agg({'Predicted Sales': 'sum'})
        temp = temp.reset_index()
        temp['Quarter Year'] = temp.apply(lambda row: f"Q{row['Quarter']} {row['Year']}", axis=1)
        temp_df.append((x, temp))

    chunk_size = 20
    temp_df = [temp_df[i:i+chunk_size] for i in range(0, len(temp_df), chunk_size)]

    for i, chunk in enumerate(temp_df):
        # plt.figure(figsize=(10, 6))
        for x in chunk:
            sns.lineplot(x[1], x='Quarter Year', y='Predicted Sales', label=x[0])
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title('Quarterly Sales Comparison Among Countries')
        plt.xlabel('Quarter Year')
        plt.ylabel('Predicted Sales (in millions)')

        ax = plt.gca()  
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_yaxis))

        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        temp = ''

        for i, c in enumerate(country_ranking):
            sales = temp1[temp1["Country"]==c]['Predicted Sales'].values[0]
            temp += f'{i+1}. {c} : {add_commas(sales)}\n'

        temp = f'''

        Country Predicted Sales Volume for the Next Year:

        {temp}


        '''
        return buffer, temp

plot, text = country_line_plot()
elements.append(Paragraph('Country Predicted Sales Report', styles['h3']))
elements.append(Paragraph(text.replace('\n', '<br />'), styles['Normal']))
elements.append(Image(plot, width=400, height=300))

# line graphs for each brand
def brand_line_plot():
    temp_df = []

    # get only top 40 brands
    temp1 = predict_df.groupby(['Brand'], as_index=False).agg({'Predicted Sales': 'sum'}).sort_values(by='Predicted Sales', ascending=False)
    brand_ranking = temp1['Brand'][:40]

    for x in brand_ranking:
        temp = predict_df[predict_df['Brand'] == x]
        temp = temp.groupby(['Quarter', 'Year']).agg({'Predicted Sales': 'sum'})
        temp = temp.reset_index()
        temp['Quarter Year'] = temp.apply(lambda row: f"Q{row['Quarter']} {row['Year']}", axis=1)
        temp_df.append((x, temp))

    chunk_size = 10
    temp_df = [temp_df[i:i+chunk_size] for i in range(0, len(temp_df), chunk_size)]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    count = 0

    for r in range(2):
        for c in range(2):
            for x in temp_df[count]:
                sns.lineplot(x[1], x='Quarter Year', y='Predicted Sales', label=x[0], ax=axs[r][c])
            
            count += 1
            axs[r][c].legend(loc='upper left', bbox_to_anchor=(1, 1))
            axs[r][c].set_title(f'Quarterly Sales Comparison (Rank {count*chunk_size-9}-{count*chunk_size})')
            axs[r][c].yaxis.set_major_formatter(ticker.FuncFormatter(format_yaxis))

    fig.suptitle('Quarterly Sales for top 40 brands Globally')
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    
    temp = ''

    for i, c in enumerate(brand_ranking):
        sales = temp1[temp1["Brand"]==c]['Predicted Sales'].values[0]
        temp += f'{i+1}. {c} : {add_commas(sales)}\n'

    temp = f'''

    Top 40 Brand (Globally) Predicted Sales Volume for the Next Year:

    {temp}


    '''
    return buffer, temp

plot, text = brand_line_plot()
elements.append(Paragraph('Glabel Brand Predicted Sales Report', styles['h3']))
elements.append(Paragraph(text.replace('\n', '<br />'), styles['Normal']))
elements.append(Image(plot, width=400, height=300))

# line graphs for each bodytype
def bodytype_line_plot():
    temp1 = predict_df.groupby(['Body Type'], as_index=False).agg({'Predicted Sales': 'sum'}).sort_values(by='Predicted Sales', ascending=False)
    bodytype_ranking = temp1['Body Type']
    temp_df = []
    for x in bodytype_ranking:
        temp = predict_df[predict_df['Body Type'] == x]
        temp = temp.groupby(['Quarter', 'Year']).agg({'Predicted Sales': 'sum'})
        temp = temp.reset_index()
        temp['Quarter Year'] = temp.apply(lambda row: f"Q{row['Quarter']} {row['Year']}", axis=1)
        temp_df.append((x, temp))



    for x in temp_df:
        sns.lineplot(x[1], x='Quarter Year', y='Predicted Sales', label=x[0])
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title('Quarterly Sales Comparison Among Body Type (Globally)')
    plt.xlabel('Quarter Year')
    plt.ylabel('Predicted Sales')

    ax = plt.gca()  
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_yaxis))

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    temp = ''


    for i, c in enumerate(bodytype_ranking):
        sales = temp1[temp1["Body Type"]==c]['Predicted Sales'].values[0]
        temp += f'{i+1}. {c} : {add_commas(sales)}\n'

    temp = f'''

    Body Type (Globally) Predicted Sales Volume for the Next Year:

    {temp}


    '''
    return buffer, temp

plot, text = bodytype_line_plot()
elements.append(Paragraph('Glabel Body Type Predicted Sales Report', styles['h3']))
elements.append(Paragraph(text.replace('\n', '<br />'), styles['Normal']))
elements.append(Image(plot, width=400, height=300))

# for each body type, top 5 selling brand
temp = predict_df.groupby(['Body Type'], as_index=False).agg({'Predicted Sales': 'sum'}).sort_values(by='Predicted Sales', ascending=False)
bodytype_ranking = temp['Body Type']

bodytype_texts = []
for bt in bodytype_ranking:
    temp = predict_df[predict_df['Body Type']==bt]
    temp = temp.groupby(['Brand']).agg({'Predicted Sales': 'sum'}).sort_values(by='Predicted Sales', ascending=False)
    temp1 = temp.reset_index()

    temp = ''
    for x in range(5):
        brand = temp1.iloc[x, 0]
        sales = temp1.iloc[x, 1]
        temp += f'{x+1}. {brand}: {add_commas(sales)}\n'

    temp = f'''
    {bt} Body Type Top 5 brands Predicted Sales Volume for the Next Year:

    {temp}


    '''
    bodytype_texts.append(temp)

elements.append(Paragraph(f'Body Type Sales Report', styles['h3']))
for t in bodytype_texts:
    elements.append(Paragraph(t.replace('\n', '<br />'), styles['Normal']))

# function for each country 
def brand_plot(country):
    temp_df = []

    # get only top 40 brands
    temp1 = predict_df[predict_df['Country']==country]
    temp1 = temp1.groupby(['Brand'], as_index=False).agg({'Predicted Sales': 'sum'}).sort_values(by='Predicted Sales', ascending=False)
    brand_ranking = temp1['Brand']

    for x in brand_ranking[:40]:
        temp = predict_df[predict_df['Brand'] == x]
        temp = temp.groupby(['Quarter', 'Year']).agg({'Predicted Sales': 'sum'})
        temp = temp.reset_index()
        temp['Quarter Year'] = temp.apply(lambda row: f"Q{row['Quarter']} {row['Year']}", axis=1)
        temp_df.append((x, temp))

    chunk_size = 10
    temp_df = [temp_df[i:i+chunk_size] for i in range(0, len(temp_df), chunk_size)]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    count = 0

    for r in range(2):
        for c in range(2):
            for x in temp_df[count]:
                sns.lineplot(x[1], x='Quarter Year', y='Predicted Sales', label=x[0], ax=axs[r][c])
            
            count += 1
            axs[r][c].legend(loc='upper left', bbox_to_anchor=(1, 1))
            axs[r][c].set_title(f'Quarterly Sales Comparison (Rank {count*chunk_size-9}-{count*chunk_size})')
            axs[r][c].yaxis.set_major_formatter(ticker.FuncFormatter(format_yaxis))

    fig.suptitle(f'Quarterly Sales for Top 40 Brands ({country})')
    fig.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    temp = ''

    for i, c in enumerate(brand_ranking[:40]):
        sales = temp1[temp1["Brand"]==c]['Predicted Sales'].values[0]
        temp += f'{i+1}. {c} : {add_commas(sales)}\n'

    temp = f'''

    Top 40 Brand ({country}) Predicted Sales Volume for the Next Year:

    {temp}


    '''
    return buffer, temp, len(brand_ranking)

def bodytype_plot(country):
    temp1 = predict_df[predict_df['Country']==country]
    temp1 = predict_df.groupby(['Body Type'], as_index=False).agg({'Predicted Sales': 'sum'}).sort_values(by='Predicted Sales', ascending=False)
    bodytype_ranking = temp1['Body Type']
    temp_df = []
    for x in bodytype_ranking:
        temp = predict_df[predict_df['Body Type'] == x]
        temp = temp.groupby(['Quarter', 'Year']).agg({'Predicted Sales': 'sum'})
        temp = temp.reset_index()
        temp['Quarter Year'] = temp.apply(lambda row: f"Q{row['Quarter']} {row['Year']}", axis=1)
        temp_df.append((x, temp))

    chunk_size = 20
    temp_df = [temp_df[i:i+chunk_size] for i in range(0, len(temp_df), chunk_size)]

    for i, chunk in enumerate(temp_df):
        # plt.figure(figsize=(10, 6))
        for x in chunk:
            sns.lineplot(x[1], x='Quarter Year', y='Predicted Sales', label=x[0])
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.title(f'Quarterly Sales for Body Type ({country})')
        plt.xlabel('Quarter Year')
        plt.ylabel('Predicted Sales')

        ax = plt.gca()  
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_yaxis))

        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()

        temp = ''

        for i, c in enumerate(bodytype_ranking):
            sales = temp1[temp1["Body Type"]==c]['Predicted Sales'].values[0]
            temp += f'{i+1}. {c} : {add_commas(sales)}\n'

        temp = f'''

        Body Type ({country}) Predicted Sales Volume for the Next Year:

        {temp}


        '''
        return buffer, temp
    
# plots and text report for each country, only top 10 countries
for c in country_ranking[:10]:
    year_predicted_sales = predict_df[predict_df['Country']==c]
    year_predicted_sales = year_predicted_sales.groupby(['Body Type', 'Brand', 'Year'], as_index=False).agg({'Predicted Sales': 'sum'})
    year_predicted_sales = year_predicted_sales.sort_values(by='Predicted Sales', ascending=False)
    year_predicted_sales['Predicted Sales'] = year_predicted_sales['Predicted Sales'].apply(add_commas)
    data_list = [year_predicted_sales.columns.values.tolist()] + year_predicted_sales.head(10).values.tolist()
    table = Table(data_list)

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
    ])

    table.setStyle(style)
    

    b_plot, b_text, no_brand = brand_plot(c)

    elements.append(Paragraph(f'{c} Sales Prediction', styles['h3']))
    elements.append(Paragraph(f"Top 10 Predicted Sales in {c}", centered_h3_style))
    elements.append(table)
    elements.append(Paragraph(f'<br /><br />There is a total of {no_brand} automotive brands being sold in {c}.<br /><br />', styles['Normal']))
    elements.append(Paragraph(b_text.replace('\n', '<br />'), styles['Normal']))

    elements.append(Image(b_plot, width=400, height=300))
    

    bt_plot, bt_text = bodytype_plot(c)
    elements.append(Paragraph(bt_text.replace('\n', '<br />'), styles['Normal']))
    elements.append(Image(bt_plot, width=400, height=300))

elements.append(Paragraph("<br /><br />End of Report", styles['Title']))

# Build the PDF
doc.build(elements)
logging.debug(f'Report generated and saved at {report_output_path}')

logging.debug("FINISHED ALGORITHM EXECUTION")