import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, PageBreak
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import matplotlib.pyplot as plt 
import seaborn as sns
from pathlib import Path
import os
import json 
from urllib import request
from io import StringIO
from urllib.parse import unquote
sns.set_theme()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s', handlers=[logging.StreamHandler()])
logging.info("Starting logging")

# LOAD FILE ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
path_input = Path(
    os.path.join(os.environ.get("INPUTS", "/data/inputs"), "algoCustomData.json")
)
path_output = Path(os.environ.get("OUTPUTS", "/data/outputs"))
path_logs = Path(os.environ.get("LOGS", "/data/logs"))

path_output_file = os.path.join(path_output, 'registration_with_risk_assed.csv')
report_output_path = os.path.join(path_output, 'report.pdf')



algoCustomData = {}

logging.debug("Loading input files...")
with open(path_input, "r") as json_file:
    algoCustomData = json.load(json_file)

result_data = algoCustomData["resultUrls"]

blacklist_df_list = []

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
                
            # load model and transformer
            if filename.lower().endswith('.pkl') and 'scaler' in filename.lower():
                scaler = joblib.load(BytesIO(response.read()))
                
            elif filename.lower().endswith('.pkl') and 'encoder' in filename.lower():
                encoder = joblib.load(BytesIO(response.read()))
                
            elif filename.lower().endswith('.joblib') and 'model' in filename.lower():
                model = joblib.load(BytesIO(response.read()))

            elif filename.lower().endswith('.csv') and 'blacklist' in filename.lower():
                csv_data = response.read().decode("utf-8")
                blacklist_df_list.append(pd.read_csv(StringIO(csv_data)))

            elif filename.lower().endswith('.csv') and 'crime' in filename.lower():
                csv_data = response.read().decode("utf-8")
                crime_df = pd.read_csv(StringIO(csv_data))

            elif filename.lower().endswith('.csv') and 'mental_health' in filename.lower():
                csv_data = response.read().decode("utf-8")
                mental_health_df = pd.read_csv(StringIO(csv_data))

            elif filename.lower().endswith('.csv') and 'registration' in filename.lower():
                csv_data = response.read().decode("utf-8")
                registration_df = pd.read_csv(StringIO(csv_data))

    except Exception as e:
        raise Exception(f"Error fetching data from URL: {url}, error: {e}")
logging.debug('Loaded all input files.')

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

logging.debug('Combing the data.')
# combine data
crime_ids = crime_df['id'].tolist()
mental_health_ids = mental_health_df['id'].to_list()
df = registration_df.copy()


# identify if person has crime and mental health records
def identify_crime_and_mental_health(row):
    if row['id'] in crime_ids:
        row['crime'] = 1
    else: row['crime'] = 0

    if row['id'] in mental_health_ids:
        row['mental_health'] = 1
    else: row['mental_health'] = 0
    
    return row
df = df.apply(identify_crime_and_mental_health, axis=1)

# count the number of blacklists the person is in
df['in_blacklist'] = 0
def identify_blacklist(row):
    if row['id'] in bl_ids:
        row['in_blacklist'] += 1
    return row

for bl in blacklist_df_list:
    bl_ids = bl['id'].tolist()
    df = df.apply(identify_blacklist, axis=1)
logging.debug('Data successfully combined.')

logging.debug('Begin scaling and encoding the data.')
# process data
df[df == 'Yes'] = 1
df[df == 'No'] = 0

education_map = {
    'Below year 10': 0,
    'Completed year 10': 1,
    'Completed year 12': 2,
    'Certificate or diploma': 3,
    'Bachelors or higher': 4
}
housing_map = {
    'Rent': 0,
    'Own with mortgage': 1,
    'Own outright': 2
}
df['education'] = df['education'].map(education_map)
df['housing_tenure'] = df['housing_tenure'].map(housing_map)


def calculate_age(row):
    row['age'] = datetime.today().year - row['birth_year']
    return row
df = df.apply(calculate_age, axis=1)

# encode and scale data
rest_data = df[['education', 'housing_tenure','smoke','alcohol','drug','debt','mental_health','crime']]
numeric_data = df[['age', 'income', 'in_blacklist']]
cat_data = df[['nationality','job_sector','relationship','household_composition']]

scaled_numeric = scaler.transform(numeric_data)
encoded_cat = encoder.transform(cat_data).toarray()
# combine train data
data = np.concatenate((scaled_numeric, rest_data.to_numpy(), encoded_cat), axis=1)

logging.debug('Data successfull scaled and encoded.')

# predict
logging.debug('Predicting risk level.')
predict = model.predict(data)
registration_df['risk level']  = predict
logging.debug('Risks Accessed.')
registration_df.to_csv(path_output_file, index=False)
logging.debug(f'Result is saved to {path_output_file}.')



##### GENERATING REPORT ######
df = registration_df.copy()
label=['Negligible', 'Minor', 'Moderate', 'Significant', 'Severe']
df['age'] = datetime.today().year - df['birth_year']
logging.debug('Generating report....')
# start pdf
doc = SimpleDocTemplate(report_output_path, pagesize=A4)
styles = getSampleStyleSheet()
centered_h3_style = ParagraphStyle(
        name='CenteredH3',
        parent=styles['Heading3'],  # Inherit properties from Heading3 style
        alignment=1,  # 0=left, 1=center, 2=right, 3=justify
    )

# nationality related
elements = []
elements.append(Paragraph("Report on Registration Risk Assessment", styles['Title']))
elements.append(PageBreak())

# start with risl proportion
x = df['risk level'].value_counts()
x = x[label]
percentages = x / x.sum() * 100
largest_index = percentages.idxmax()
explode = [0.1 if idx == largest_index else 0 for idx in x.index]

wedges, texts, autotexts = plt.pie(x, labels=x.index, autopct='%1.1f%%', explode=explode, shadow=True, startangle=90,
        textprops={'size': 'smaller'}, radius=0.8)
plt.legend(title="Risk Level", loc="upper left", bbox_to_anchor=(0.9, 0, 0, 1.1))
plt.title('Proportion of the Risk Level')
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
plt.close()
text = f'''
Risk Level Count
{x.index[0]}: {x.values[0]}
{x.index[1]}: {x.values[1]}
{x.index[2]}: {x.values[2]}
{x.index[3]}: {x.values[3]}
{x.index[4]}: {x.values[4]}
'''
elements.append(Paragraph(text.replace('\n', '<br />'), styles['Normal']))
elements.append(Image(buffer, width=350, height=250))


# add nationality related information
x = df['nationality'].value_counts()
text = f'''
There are a total of {df['nationality'].nunique()} nationalities. The top 5 nationalities with the most registration are:
1. {x.index[0]} with {x.values[0]} registrations
2. {x.index[1]} with {x.values[1]} registrations
3. {x.index[2]} with {x.values[2]} registrations
4. {x.index[3]} with {x.values[3]} registrations
5. {x.index[4]} with {x.values[4]} registrations


'''

pivot_table = df[df['nationality'].isin(x.index[:5])].pivot_table(
    index='nationality', columns='risk level', aggfunc='size', fill_value=0
    )

# Create a table
pivot_table = pivot_table[label]
pivot_data = [pivot_table.columns.to_list()] + pivot_table.reset_index().values.tolist()
pivot_data[0].insert(0, 'Nationality')
table = Table(pivot_data)

# Apply styles to the table
style = TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ('BACKGROUND', (0, 1), (0, 5), colors.lightblue), 
    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 1), (-1, -1), 12),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ('TOPPADDING', (0, 1), (-1, -1), 6),
])

table.setStyle(style)

# heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table[:5], annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap: Nationality vs. Risk Level')
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
plt.close()

elements.append(Paragraph('Nationality Related', styles['h3']))
elements.append(Paragraph(text.replace('\n', '<br />'), styles['Normal']))

elements.append(Paragraph("Count Table of the Top 5 Nationality", centered_h3_style))
elements.append(table)

elements.append(PageBreak())
elements.append(Paragraph("Heatmap", centered_h3_style))
elements.append(Image(buffer, width=350, height=250))



# age related
sns.kdeplot(data=df, x='age', hue='risk level', hue_order=label, fill=True)
plt.title('Density Plot on Age')
plt.xlabel('Age')
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
plt.close()

text = f'The age range is between {df["age"].min()} and {df["age"].max()}.'

elements.append(Paragraph('Age Related', styles['h3']))
elements.append(Paragraph(text.replace('\n', '<br />'), styles['Normal']))
elements.append(Image(buffer, width=400, height=300))


# income related
sns.kdeplot(data=df, x='income', hue='risk level', hue_order=label, fill=True)
plt.title('Density Plot on Income')
plt.xlabel('Income')
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
plt.close()

text = f'The income range is between US${df["income"].min()} and US${df["income"].max()}.'

elements.append(PageBreak())
elements.append(Paragraph('Income Related', styles['h3']))
elements.append(Paragraph(text.replace('\n', '<br />'), styles['Normal']))
elements.append(Image(buffer, width=400, height=300))

sns.scatterplot(data=df, x='age', y='income', hue='risk level', hue_order=label)
plt.legend(title="Risk Level")
plt.ylabel('Income')
plt.xlabel('Age')
plt.title('Scatter Plot: Age angaist Income')
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
plt.close()
elements.append(Image(buffer, width=400, height=300))


# job sector related
sns.countplot(data=df, y = 'job_sector', hue='risk level', hue_order=label)
plt.legend(title='Risk Level')
plt.title('Count Plot: Job Sector & Risk Level')
plt.ylabel('Job Sector')
plt.xlabel('Count')
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
plt.close()

x = df['job_sector'].value_counts()
text = f'''
There are a total of {df['job_sector'].nunique()} job sectors. The top 3 job sectors are:
1. {x.index[0]} with {x.values[0]} registrations
2. {x.index[1]} with {x.values[1]} registrations
3. {x.index[2]} with {x.values[2]} registrations


'''
elements.append(PageBreak())
elements.append(Paragraph('Job Sector Related', styles['h3']))
elements.append(Paragraph(text.replace('\n', '<br />'), styles['Normal']))
elements.append(Image(buffer, width=400, height=350))


# education related
sns.countplot(data=df, y = 'education', hue='risk level', hue_order=label, 
              order=['Bachelors or higher',
                     'Certificate or diploma',
                     'Completed year 12',
                     'Completed year 10',
                     'Below year 10'])
plt.legend(title='Risk Level')
plt.title('Count Plot: Education & Risk Level')
plt.ylabel('Education')
plt.xlabel('Count')
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
plt.close()

x = df['education'].value_counts()
text = f'''
There are a total of {df['education'].nunique()} job sectors. The top 3 job sectors are:
1. {x.index[0]} with {x.values[0]} registrations
2. {x.index[1]} with {x.values[1]} registrations
3. {x.index[2]} with {x.values[2]} registrations


'''
elements.append(Paragraph('Education Related', styles['h3']))
elements.append(Paragraph(text.replace('\n', '<br />'), styles['Normal']))
elements.append(Image(buffer, width=400, height=300))


# relationship related 
sns.countplot(data=df, x = 'relationship', hue='risk level', hue_order=label)
plt.legend(title='Risk Level')
plt.title('Count Plot: Relationship & Risk Level')
plt.xlabel('Relation')
plt.ylabel('Count')
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
plt.close()
elements.append(Paragraph('Relationship Related', styles['h3']))
elements.append(Image(buffer, width=400, height=300))

# household related
sns.countplot(data=df, y = 'household_composition', hue='risk level', hue_order=label)
plt.legend(title='Risk Level')
plt.title('Count Plot: Household Composition & Risk Level')
plt.xlabel('Count')
plt.ylabel('Household Composition')
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
plt.close()
elements.append(PageBreak())
elements.append(Paragraph('Household Composition Related', styles['h3']))
elements.append(Image(buffer, width=400, height=300))


# housing tenure related
sns.countplot(data=df, x = 'housing_tenure', hue='risk level', hue_order=label)
plt.legend(title='Risk Level')
plt.title('Count Plot: Housing Tenure & Risk Level')
plt.xlabel('Housing Tenure')
plt.ylabel('Count')
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
plt.close()
elements.append(Paragraph('Housing Tenure Related', styles['h3']))
elements.append(Image(buffer, width=400, height=300))



# other proportional
x = df['gender'].value_counts()
explode = [0.1, 0]
plt.pie(x, labels=x.index, autopct='%1.1f%%', explode=explode, shadow=True, startangle=90,
        textprops={'size': 'smaller'}, radius=0.8)
plt.legend(title="Gender", loc="upper left", bbox_to_anchor=(0.9, 0, 0, 1.1))
plt.title('Proportion of the Gender')
plt.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
plt.close()
elements.append(PageBreak())
elements.append(Paragraph('Other Proportional Related', styles['h3']))
elements.append(Image(buffer, width=400, height=300))


fig, axs = plt.subplots(2, 2)
explode=[0.1, 0]
fields = ['smoke','alcohol','drug','debt']
titles = [ 'Proportion of Smoker', 
        'Proportion of Chronic Drinke', 'Proportion of Drug History', 'Proportion of Debtor']
legends = ['Smoker', 'Chronic Drinker', 'Drug History', 'Debtor']
count = 0
for i in range(2):
    for p in range(2):
        
        x = df[fields[count]].value_counts()
        axs[i, p].pie(x, labels=x.index, autopct='%1.1f%%', explode=explode, shadow=True, startangle=90,
        textprops={'size': 'smaller'}, radius=0.8)
        axs[i, p].set_title(titles[count])

        axs[i, p].legend(title=legends[count], loc="upper left", bbox_to_anchor=(1, 0, 0, 1))
        count+=1
        if count == 5:
                break
fig.suptitle('Proportional Insights')
fig.tight_layout()
buffer = BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
plt.close()
elements.append(Image(buffer, width=400, height=300))


# Build the PDF
doc.build(elements)
logging.debug(f'Report generated and saved at {report_output_path}')

logging.debug("FINISHED ALGORITHM EXECUTION")
