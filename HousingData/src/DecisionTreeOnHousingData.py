#!/usr/bin/python

'''
Created on July 11, 2017
@author: Dhara
'''

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

from DecisionTreeClassifier import DecisionTreeClassifier

class BostonHousingData(object):
    """
    Class dealing with reading Boston housing data and predicting
    """
    FEATURES = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD",
                    "TAX", "PTRATIO", "B", "LSTAT"]

    def __init__(self):
        pass

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

    def read_data(self, file_path_of_data):
        """
        Reads data from tab separated text file into numpy array
        @type file_path_of_data: string
        """
        return self.normalize_data(
            self.read_data_from_data_frame(
                self.read_data_frame_from_text_file(file_path_of_data)))

    def normalize_data(self, data):
        """
        For each column/features do following
        Find min(feature=f) and max(feaure=f)
        (data[f] - min[f])/max[f]
        """
        return (data - np.amin(data, axis=0))/np.amax(data, axis=0)

    def rmse(self, y_predicted, y_actual):
        return sqrt(mean_squared_error(y_actual, y_predicted))

def main():
    train_path='../data/housing_train.txt'
    test_path = '../data/housing_test.txt'
    boston_housing_data = BostonHousingData()
    arr = boston_housing_data.read_data(train_path)
    x = np.delete(arr, -1, axis=1) #Remove labels
    y = np.array([arr[:,-1]]).T #To convert Y from mx (array), to m x 1 (matrix)
    classifier = DecisionTreeClassifier()
    classifier.train(x, y)

    test_arr = boston_housing_data.read_data(test_path)
    test_x = np.delete(test_arr, -1, axis=1)
    test_y = np.array([test_arr[:,-1]]).T
    predicted_y = classifier.predict(test_x)
    print(boston_housing_data.rmse(predicted_y, test_y))
main()
