# Libraries ----------------------------------------
import re
import pandas as pd
import os
import sys

sys.path.append(os.getcwd())
# Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

# Loading data ----------------------------------------

df = pd.read_csv("files/datasets/intermediate/a02_merged_df.csv")

# null data filling ----------------------------------------

for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"]:
    df[col] = df[col].fillna("No")

df["InternetService"] = df["InternetService"].fillna("No Service")

print(df.isnull().sum())

# creation of new columns ----------------

df['exited'] = (df['EndDate'] != 'No').astype(int)

df.loc[df['EndDate'] == 'No', 'EndDate'] = '2020-02-01 00:00:00'
df['BeginDate'] = pd.to_datetime(df['BeginDate'], format='%Y-%m-%d')
df['EndDate'] = pd.to_datetime(df['EndDate'], format='%Y-%m-%d %H:%M:%S')
extraction_date = pd.to_datetime('2020-02-01 00:00:00')
df['days_since_join'] = (extraction_date - df['BeginDate']).dt.days


print(df.head(10))

#  coding variables ----------------

for col in df.columns:
    if set(df[col].unique()) == {"No", "Yes"}:
        df[col] = (df[col] == "Yes").astype(int)


df["gender"] = (df["gender"] == "Male").astype("int")


# transformation snake case ----------------

def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


df.columns = [to_snake_case(col) for col in df.columns]

print(df.head(10))


# Save data ----------------------------------------

df.to_csv("files/datasets/intermediate/a03_preprocessing.csv", index=False)
