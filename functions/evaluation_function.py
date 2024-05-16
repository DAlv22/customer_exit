import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import os
import sys
from sklearn.model_selection import cross_val_predict

sys.path.append(os.getcwd())

def evaluate_model(model, train_features, train_target, test_features, test_target):
    
    eval_stats = {}
    
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):
        eval_stats[type] = {}
        
        if type == 'train':
            color = 'blue'
            data = train_features
            true_target = train_target
        else:
            color = 'green'
            data = test_features
            true_target = test_target
    
        pred_proba = cross_val_predict(model, data, true_target, cv=5, method='predict_proba')
        
        
        fpr, tpr, roc_thresholds = metrics.roc_curve(true_target, pred_proba)
        roc_auc = metrics.roc_auc_score(true_target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc


        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'            
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])    
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')        
        ax.set_title(f'ROC Curve')
        
        pred_target = cross_val_predict(model, data, true_target, cv=5)
        
        eval_stats[type]['Accuracy'] = metrics.accuracy_score(true_target, pred_target)
    
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'ROC AUC'))
    
    print(df_eval_stats)
    
    return df_eval_stats