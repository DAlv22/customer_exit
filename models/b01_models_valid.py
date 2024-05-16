# Libraries ----------------------------------------

import os
import sys
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from lightgbm import LGBMClassifier

sys.path.append(os.getcwd())
# Esto es para agregar al path la ruta de ejecución actual y poder importar respecto a la ruta del proyecto, desde donde se debe ejecutar el código
from functions.evaluation_function import evaluate_model

# Loading data ----------------------------------------

train_features = pd.read_csv(
    "files/datasets/intermediate/a06_train_features.csv")
train_target = pd.read_csv("files/datasets/intermediate/a06_train_target.csv")

test_target = pd.read_csv("files/datasets/intermediate/a06_test_target.csv")
test_features = pd.read_csv(
    "files/datasets/intermediate/a06_test_features.csv")


# ---------------- sanity model ----------------

base_model = DummyClassifier(
    strategy='constant', constant=1, random_state=12345)
base_model.fit(train_features, train_target)

result = evaluate_model(base_model, train_features,
                        train_target, test_features, test_target)


# ---------------- Logistic Regression ----------------

lr_model = LogisticRegression(random_state=12345, class_weight='balanced')
lr_model.fit(train_features, train_target)

result = evaluate_model(lr_model, train_features,
                        train_target, test_features, test_target)


# ---------------- Random Forest ----------------
rfc = RandomForestClassifier(class_weight='balanced')

params = {
    "n_estimators": [500, 700],
    "max_depth": [6, 7, 8, 9, 10]
}

gsSVR = GridSearchCV(estimator=rfc, param_grid=params,
                     n_jobs=-1, verbose=0, scoring='roc_auc')

gsSVR.fit(train_features, train_target)

result = evaluate_model(gsSVR, train_features,
                        train_target, test_features, test_target)

# ---------------- LGBM Classifier ----------------
lgbm = LGBMClassifier(random_state=12345, class_weight='balanced')
lgbm.fit(train_features, train_target)

result = evaluate_model(lgbm, train_features,
                        train_target, test_features, test_target)

# ---------------- CatBoost ----------------
CB = CatBoostClassifier(random_state=12345, verbose=0)
CB.fit(train_features, train_target)

result = evaluate_model(CB, train_features, train_target,
                        test_features, test_target)
