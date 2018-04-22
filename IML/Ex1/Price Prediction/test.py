import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data = np.genfromtxt("kc_house_data.csv", dtype=float, delimiter=',', names=True, invalid_raise = False)

dataset = pd.read_csv ("kc_house_data.csv")
dataset.iloc[20672]


# Preprocessing
def missing_columns(data):
    """
    Finds and removes any rows where there are NaN columns.
    Returns the dataframe without these rows.
    """
    missing_rows_indices = data.notnull().all(axis=1)
    return data[missing_rows_indices].copy()


def out_of_range_values(data):
    impossible_negative_cols = ["bedrooms", "bathrooms", "waterfront",
                                "view", "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
                                "zipcode", "sqft_living15", "sqft_lot15"]
    impossible_zero_cols = ["price", "sqft_living", "sqft_lot", "floors", "condition", "grade"]
    negative_values_filter = (data[impossible_negative_cols] >= 0).all(axis=1)
    zero_values_filter = (data[impossible_zero_cols] > 0).all(axis=1)
    return data[negative_values_filter & zero_values_filter]


def remove_id_column(data):
    return data.drop('id', 1).copy()


def add_interceptor_feature(data):
    new_data = data.copy()
    new_data["interceptor"] = 1
    return new_data

def convert_categorical_features(data):
    data.zipcode = data.zipcode.astype('category')


print(dataset.shape)
dataset = missing_columns(dataset)
print(dataset.shape)
dataset = out_of_range_values(dataset)
print(dataset.shape)
dataset = remove_id_column(dataset)
print(dataset.shape)
dataset = add_interceptor_feature(dataset)
print (dataset.shape)
print (dataset.dtypes)
convert_categorical_features(dataset)
print (dataset.dtypes)
print (dataset)
