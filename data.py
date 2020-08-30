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

    

    

