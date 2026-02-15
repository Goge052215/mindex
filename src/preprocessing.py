"""
preprocess the data
    1. drop the columns with high missing values
    2. fill the missing values with the mean
    3. encode the categorical variables

@goge052215
"""

import pandas as pd


class Preprocessor:
    def __init__(self, df):
        self.df = df
    
    # Drop columns that are not useful (ID and raw text)
    def drop_unuseful_columns(self):
        self.df = self.df.drop(['textID', 'text', 'selected_text'], axis=1, errors='ignore')
    
    # Encode the categorical variables
    def encode_categorical_variables(self):
        self.df['sentiment'] = self.df['sentiment'].map({
            'neutral': 0, 
            'negative': 1, 
            'positive': 2
        })
        
        # Encode other categorical columns
        categorical_cols = ['Time of Tweet', 'Age of User', 'Country']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category').cat.codes

    # Preprocess the data
    def preprocess(self):
        self.drop_unuseful_columns()
        self.encode_categorical_variables()
        self.df = self.df.dropna()
