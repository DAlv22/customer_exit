# Libraries ----------------------------------------

import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split

sys.path.append(os.getcwd())  
# Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

# Loading data ----------------------------------------

target=pd.read_csv("files/datasets/intermediate/a05_target.csv")
features=pd.read_csv("files/datasets/intermediate/a05_features.csv")

# Split data ----------------------------------------

train_features, temp_features, train_target, temp_target = train_test_split(
    features, target, test_size=0.40, random_state=12345)

valid_features, test_features, valid_target, test_target = train_test_split(
    temp_features, temp_target, test_size=0.50, random_state=12345)



# Save data ----------------------------------------

train_target.to_csv("files/datasets/intermediate/a06_train_target.csv", index=False)
test_target.to_csv("files/datasets/intermediate/a06_test_target.csv", index=False)
valid_target.to_csv("files/datasets/intermediate/a06_valid_target.csv", index=False)

train_features.to_csv("files/datasets/intermediate/a06_train_features.csv", index=False)
test_features.to_csv("files/datasets/intermediate/a06_test_features.csv", index=False)
valid_features.to_csv("files/datasets/intermediate/a06_valid_features.csv", index=False)