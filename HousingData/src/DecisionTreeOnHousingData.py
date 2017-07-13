#!/usr/bin/python

'''
Created on July 11, 2017
@author: Dhara
'''

import numpy as np
import pandas as pd

class BostonHousingData(object):
    """
    Class dealing with reading Boston housing data and predicting
    """
    FEATURES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
                    "TAX", "PTRATIO", "B", "LSTAT"]

    def __init__(self, train_path='../data/housing_train.txt'):
        self.train_path = train_path

    def read_data_frame_from_text_file(self, filepath):
        """
        Reads data from tab separated text file into dataframe
        @type filepath: String
        @type features: list[String]
        @return dataframe
        """
        df = pd.read_table(filepath, delim_whitespace=True, names=BostonHousingData.FEATURES)
        return df

    def read_data_from_data_frame(self, df):
        """
        Reads dataframe and returns numpy array
        @type df: data frame
        @return np.array()
        """
        return df.as_matrix()

    def read_data(self):
        """
        Reads data from tab separated text file into numpy array
        """
        return self.read_data_from_data_frame(self.read_data_frame_from_text_file(self.train_path))

    def normalize_data(self, data):
        """
        For each column/features do following
        Find min(feature=f) and max(feaure=f)
        (data[f] - min[f])/max[f]
        """
        return (A - np.amin(A, axis=0))/np.amax(A, axis=0)

def main():
    boston_housing_data = BostonHousingData()
    arr = boston_housing_data.read_data()
main()
