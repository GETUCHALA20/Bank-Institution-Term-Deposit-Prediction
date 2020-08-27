import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Preprocess:
    def __init__(self):
        self.data = pd.read_csv("data/bank-additional-full.csv", sep=";")
    
    def get_data(self):
        return self.data

    def find_outliers_tukey(self,x):
        q1 = x.quantile(.25)
        q3 = x.quantile(.75)
        iqr = q3 - q1
        floor = q1 - 1.5*iqr
        ceiling = q3 + 1.5*iqr
        outlier_indices = list(x.index[(x < floor) | (x > ceiling)])
        outlier_values = list(x[outlier_indices])
        return outlier_indices, outlier_values
    
    def handle_invalid_data(self,df,column_list):
        for column in column_list:
            df = df[df[column] != 'unknown']
        return df
    
    def integer_encoding(self,df,column,level_mapping):
        df[column] = df[column].replace(level_mapping)
        return df
    
    def one_hot_encoder(self, df, column_list):
        """Takes in a dataframe and a list of columns
        for pre-processing via one hot encoding returns
        a dataframe of one hot encoded values"""
        df_to_encode = df[column_list]
        df = pd.get_dummies(df_to_encode,drop_first=True)
        return df
    
    def label_encoding(self,df,column_list):
        """Takes in a dataframe and a list of columns
        for pre-processing via label encoding returns
        a dataframe of label encoded values"""
        encoder = LabelEncoder()
        for column in column_list:
            df[column] = encoder.fit_transform(df[column])
        return df
    
    def scale_data(self, df, column_list):
        """Takes in a dataframe and a list of column names to transform
        returns a dataframe of scaled values"""
        df_to_scale = df[column_list]
        x = df_to_scale.values
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_to_scale = pd.DataFrame(x_scaled, columns=df_to_scale.columns)
        return df_to_scale
    

if __name__ == '__main__':
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

    X = data[features]
    y = data[['y']]
    columns_to_scale= ['age','campaign','cons.conf.idx','cons.price.idx','duration','emp.var.rate','euribor3m','nr.employed',
                   'pdays','previous']
    X = p.scale_data(X,columns_to_scale)
    

