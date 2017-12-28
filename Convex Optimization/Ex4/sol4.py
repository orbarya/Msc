import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def least_squares(X, y):
    theta = np.linalg.inv(X.T*X) * X.T * y
    return theta


def output(method, X, y ,w, y_pred):
    print("Coefficients: ", w)
    print("precision: ", np.linalg.norm(y - y_pred))
    print("R_Squared: ", r2_score(y, y_pred))


def read_data():
    data = pd.read_csv('cal_housing.data')[
        ['longitude', 'latitude', 'housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households',
         'medianIncome', 'medianHouseValue']]

    X = np.asmatrix(data.drop('medianHouseValue', axis=1))
    y = np.asmatrix(data['medianHouseValue']).T
    return data, X, y

# print("X (input) dimensions: ", X)
# print("y (output) dimensions: ", y)


def add_variables(data):
    data['populationPerBedroom']=data['population']/data['totalBedrooms']
    X = np.asmatrix(data.drop('medianHouseValue', axis=1))
    y = np.asmatrix(data['medianHouseValue']).T
    return data, X, y


def print_full(x):
    print(x.to_string())

data, X, y = read_data()
print_full (data)
w = least_squares(X, y)
y_pred = X*w
output("least_squares", X, y, w, y_pred)
data, X, y = add_variables(data)
w = least_squares(X, y)
y_pred = X*w
output("least_squares_added_variables", X, y, w, y_pred)

