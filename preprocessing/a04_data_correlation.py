# Libraries ----------------------------------------

import pandas as pd
import os
import sys
from scipy.stats import chi2_contingency, pointbiserialr

sys.path.append(os.getcwd())
# Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código

# Loading data ----------------------------------------

df = pd.read_csv('files/datasets/intermediate/a03_preprocessing.csv')

# Separation of variables ----------------------------------------

categorical_variables = ['type', 'payment_method', 'internet_service']
binary_variables = ['paperless_billing', 'online_security', 'device_protection',
                    'tech_support', 'streaming_tv', 'streaming_movies',
                    'gender', 'senior_citizen', 'partner', 'dependents',
                    'multiple_lines', 'online_backup']

# categorical variables --------------------------------
for cat_var in categorical_variables + binary_variables:
    contingency_table = pd.crosstab(df[cat_var], df['exited'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(
        f"Chi-square test for {cat_var} vs. exited: P-Value= {p:.3f}")


# Numerical variables
numeric_variables = ['monthly_charges', 'total_charges', 'days_since_join']

for num_var in numeric_variables:
    r, p = pointbiserialr(df['exited'], df[num_var])
    print(
        f"Point-biserial correlation for {num_var} vs. exited: r = {r:.3f}, P-Value= {p:.3f}")
