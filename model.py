import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Linear Regression
# define the pipeline
steps = [('pca', PCA(n_components=25)), ('m', LogisticRegression())]
lr = Pipeline(steps=steps)
# evaluate model
cv = KFold(n_splits=10, random_state=45)
n_scores = cross_val_score(lr, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
scores = ['accuracy','f1','precision','recall','roc_auc']
lr_scores =cross_validate(lr, X, y, scoring=scores, cv=cv, n_jobs=-1, error_score='raise') 
# report performance
print('Accuracy: %.3f ' % (np.mean(lr_scores['test_accuracy'])))
print('F1 Score: %.3f ' % (np.mean(lr_scores['test_f1'])))
print('Precision: %.3f ' % (np.mean(lr_scores['test_precision'])))
print('Recall: %.3f ' % (np.mean(lr_scores['test_recall'])))
print('ROC AUC score: %.3f ' % (np.mean(lr_scores['test_roc_auc'])))

# XGBoost
# define the pipeline
steps = [('pca', PCA(n_components=25)), ('m', XGBClassifier())]
xgb = Pipeline(steps=steps)
# evaluate model
cv = KFold(n_splits=10, random_state=45)
n_scores = cross_val_score(xgb, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
scores = ['accuracy','f1','precision','recall','roc_auc']
xgb_scores =cross_validate(xgb, X, y, scoring=scores, cv=cv, n_jobs=-1, error_score='raise') 
# report performance
print('Accuracy: %.3f ' % (np.mean(xgb_scores['test_accuracy'])))
print('F1 Score: %.3f ' % (np.mean(xgb_scores['test_f1'])))
print('Precision: %.3f ' % (np.mean(xgb_scores['test_precision'])))
print('Recall: %.3f ' % (np.mean(xgb_scores['test_recall'])))
print('ROC AUC score: %.3f ' % (np.mean(xgb_scores['test_roc_auc'])))

# Multi-Layer Perceptron

# define the pipeline
steps = [('pca', PCA(n_components=30)), ('m', MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu',
                                                            solver='adam', max_iter=500))]
mlp = Pipeline(steps=steps)
# evaluate model
cv = KFold(n_splits=10, random_state=45)
scores = ['accuracy','f1','precision','recall','roc_auc']
mlp_scores =cross_validate(mlp, X, y, scoring=scores, cv=cv, n_jobs=-1, error_score='raise') 
# report performance
print('Accuracy: %.3f ' % (np.mean(mlp_scores['test_accuracy'])))
print('F1 Score: %.3f ' % (np.mean(mlp_scores['test_f1'])))
print('Precision: %.3f ' % (np.mean(mlp_scores['test_precision'])))
print('Recall: %.3f ' % (np.mean(mlp_scores['test_recall'])))
print('ROC AUC score: %.3f ' % (np.mean(mlp_scores['test_roc_auc'])))
