# Libraries ----------------------------------------

import pandas as pd
import os
import sys

sys.path.append(os.getcwd())
# Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

# Loading data ----------------------------------------

contract = pd.read_csv("datasets/input/contract.csv")
internet = pd.read_csv("datasets/input/internet.csv")
personal = pd.read_csv("datasets/input/personal.csv")
phone = pd.read_csv("datasets/input/phone.csv")

# Dataframes fusion ----------------------------------------

df = contract.merge(internet, how='left', on='customerID')
df = df.merge(personal, how='left', on='customerID')
df = df.merge(phone, how='left', on='customerID')

# Check data ----------------------------------------

print(df.info())
print(df.isnull().sum())

# Column 'TotalCharges'----------------------------------------

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df = df.drop(df[df['TotalCharges'].isna()].index)
df = df.reset_index(drop=True)

# Check data ----------------------------------------

print(df.info())
print(df.isnull().sum())


# Save data ----------------------------------------

df.to_csv("files/datasets/intermediate/a02_merged_df.csv", index=False)
