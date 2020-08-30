import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from data import Preprocess
p = Preprocess()
data = p.get_data()
#Dropping cons.price.index outliers
cons_index, cons_value = p.find_outliers_tukey(data["cons.price.idx"])
data = data.drop(cons_index)
#replacing duration outliers with maximum value
max_ = data['duration'].max()
data['duration'] = np.where(data.duration > 645,max_,data['duration'])

#replacing campaign outliers with maximum value
max_ = data['campaign'].max()
data['campaign'] = np.where(data.campaign > 7, max_,data['campaign'])

#handling invalid data
invalid_data = ['job','education','loan','housing','default','marital']
data = p.handle_invalid_data(data,invalid_data)

#integer encoding
level_mapping = {'illiterate': 0, 'basic.4y': 1, 'basic.6y': 2, 'basic.9y':3, 'high.school':4,'professional.course':5,
                'university.degree': 6}
data = p.integer_encoding(data,'education',level_mapping)

#label encoding 
encode_list = ['y','default','housing','loan','contact']
data = p.label_encoding(data,encode_list)

#one hot encoding 
columns=['job','marital','month','day_of_week','poutcome']
data = p.one_hot_encoder(data,columns)

features = ['age', 'education', 'default', 'housing', 'loan', 'contact', 'duration',
    'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx',
    'cons.conf.idx', 'euribor3m', 'nr.employed','job_blue-collar',
    'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired',
    'job_self-employed', 'job_services', 'job_student', 'job_technician',
    'job_unemployed', 'marital_married', 'marital_single', 'month_aug',
    'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may',
    'month_nov', 'month_oct', 'month_sep', 'day_of_week_mon',
    'day_of_week_thu', 'day_of_week_tue', 'day_of_week_wed',
    'poutcome_nonexistent', 'poutcome_success']

X = data.drop('y',axis=1)
y = data[['y']]
columns_to_scale= ['age','campaign','cons.conf.idx','cons.price.idx','duration','emp.var.rate','euribor3m','nr.employed',
                'pdays','previous']
X = p.scale_data(X,columns_to_scale)

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



