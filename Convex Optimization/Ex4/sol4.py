import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def least_squares(X, y):
    theta = np.linalg.inv(X.T*X) * X.T * y
    return theta


def output(method, X, y ,w, y_pred):
    print('------------------------------------')
    print(method)
    print("Coefficients: ", w)
    print("precision: ", np.linalg.norm(y - y_pred))
    print("R_Squared: ", r2_score(y, y_pred))
    print('------------------------------------')


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
    data['free_variables'] = 1

    data['populationPerBedroom'] = data['population']/data['totalBedrooms']
    data['incomePerPerson'] = data['medianIncome']/data['population']

    """
    
    """
    data['populationPerHousehold'] = data['population'] / data['households']
    data['populationPerHousehold'] = data['totalBedrooms'] / data['households']

    """
    longitudeIncome - Relation between income and longitude.
    latitudeIncome  - Relation between income and latitude.
    """
    data["medianIncomeSquared"] = data['medianIncome']*data['medianIncome']
    data['longitudeIncome'] = data['medianIncome'] / data['longitude']
    data['latitudeIncome'] = data['medianIncome'] / data['latitude']
    X = np.asmatrix(data.drop('medianHouseValue', axis=1))
    y = np.asmatrix(data['medianHouseValue']).T
    print (data)
    return data, X, y


#def backtracking(f,g,x,alpha=0.5, beta=0.5, t=1):


data, X, y = read_data()
w = least_squares(X, y)
y_pred = X*w
output("least_squares", X, y, w, y_pred)
data, X, y = add_variables(data)
w = least_squares(X, y)
y_pred = X*w
output("least_squares_added_variables", X, y, w, y_pred)

