# Libraries ----------------------------------------

import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler

sys.path.append(os.getcwd())
# Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

# Loading data ----------------------------------------

df = pd.read_csv("files/datasets/intermediate/a03_preprocessing.csv")

# removal of non-relevant columns --------------------------------

df = df.drop(['customer_id', 'begin_date', 'end_date', 'gender'], axis=1)

# standardisation of numerical characteristics --------------------------------

numeric_cols = ["total_charges", "monthly_charges", "days_since_join"]
scaler = StandardScaler()
scaler.fit(df[numeric_cols])

df[numeric_cols] = scaler.transform(df[numeric_cols])

# features creation ----------------------------------------

features = df.drop('exited', axis=1)

# target creation ----------------------------------------

target = df['exited']

# OHE --------------------------------

features = pd.get_dummies(features, drop_first=True)

print(features.info())

print(target)

# Save data ----------------------------------------

target.to_csv("files/datasets/intermediate/a05_target.csv", index=False)
features.to_csv("files/datasets/intermediate/a05_features.csv", index=False)
