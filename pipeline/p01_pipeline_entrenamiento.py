# Libraries ----------------------------------------

import os, sys
import argparse

sys.path.append(os.getcwd()) # Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

import params as params
    
# Defining executable file extensions ---------------------------------------- 

if params.sistema_operativo == 'Windows':
        extension_binarios = ".exe"
else:
        extension_binarios = ""

# Info ---------------------------------------- 

print(f"---------------------------------- \nComenzando proceso \n----------------------------------")

# Preproceso ---------------------------------------- 

print(f"---------------------------------- \nmerge data frames \n----------------------------------")
os.system(f"python{extension_binarios} preprocessing/a02_merge_dataframe.py") # Se ejecuta el script que renombra las columnas

print(f"---------------------------------- \nPreproceso \n----------------------------------")
os.system(f"python{extension_binarios} preprocessing\a03_preprocessing.py") # Se ejecuta el script que cambia los tipos de datos

print(f"---------------------------------- \nFData correlation \n----------------------------------")
os.system(f"python{extension_binarios} preprocessing\a04_data_correlation.py") # Se ejecuta el script que fusiona los dataframes

print(f"---------------------------------- \nremoval and standardisation \n----------------------------------")
os.system(f"python{extension_binarios} preprocessing\a05_removal_and_standardisation_of_columns.py") # Se ejecuta el script que crea las características y el objetivo

print(f"---------------------------------- \ndata separation \n----------------------------------")
os.system(f"python{extension_binarios} preprocessing\a06_data_separation.py") # Se ejecuta el script que separa los datos en entrenamiento y testeo


# Function

print(f"---------------------------------- \nevaluation function \n----------------------------------")
os.system(f"python{extension_binarios} functions\evaluation_function.py") # Se ejecuta la función de evaluacion

# Modelo ---------------------------------------- 
