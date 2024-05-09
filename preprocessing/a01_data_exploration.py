# Libraries ----------------------------------------

import pandas as pd
import os
import sys

sys.path.append(
    os.getcwd()
)  # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

# Loading data ----------------------------------------

contract = pd.read_csv("datasets/input/contract.csv")
internet = pd.read_csv("datasets/input/internet.csv")
personal = pd.read_csv("datasets/input/personal.csv")
phone = pd.read_csv("datasets/input/phone.csv")

# Data exploration ----------------------------------------
#  function for single values per column


def unique_values(dataframe):
    for column in dataframe.columns:
        unique_values = dataframe[column].unique()
        print(f"Valores unicos en la columna '{column}':")
        print(unique_values)
        print()


#  --------- contract ---------
# General information
contract.shape
contract.info()
contract.describe()
unique_values(contract)
# Null values
contract.isnull().sum()
# duplicate values
contract.duplicated().sum()

#  --------- internet ---------
# General information
internet.shape
internet.info()
internet.describe()
unique_values(internet)
# Null values
internet.isnull().sum()
# duplicate values
internet.duplicated().sum()

#  --------- personal ---------
# General information
personal.shape
personal.info()
personal.describe()
unique_values(personal)
# Null values
personal.isnull().sum()
# duplicate values
personal.duplicated().sum()

#  --------- phone ---------
# General information
phone.shape
phone.info()
phone.describe()
unique_values(phone)
# Null values
phone.isnull().sum()
# duplicate values
phone.duplicated().sum()
