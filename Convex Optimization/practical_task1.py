import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data = np.genfromtxt("kc_house_data.csv", dtype=float, delimiter=',', names=True, invalid_raise = False)


def convert_date(data):
    new_data = data.copy()
    date = new_data.columns.get_loc('date')
    for i in range(0, data.shape[0]):
        print (str(i) + " " +  new_data['date'].iloc[i])
        new_data.iloc[i, date] = int((new_data.iloc[i,date]).split("T")[0])
    return new_data

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
    new_data = data.copy()
    zipcode_one_hot = pd.get_dummies(new_data['zipcode'])
    new_data.drop('zipcode', 1)
    new_data.join(zipcode_one_hot)
    return new_data


def remove_langtitude_longtitude(data):
    new_data = data.copy()
    new_data = new_data.drop('lat', 1)
    new_data = new_data.drop('long', 1)
    return new_data


def split_train_test(X, y, x):
    """
    Splits the dataset and label randomly with a percentage in favor of the train set given by x.
    :param X: Features dataset.
    :param y: Labels
    :param x: Percentage of examples in favor of the training set.
    :return:
    """
    mask = np.random.rand(len(X)) < (x / 100)
    return X[mask], y[mask], X[~mask], y[~mask]


def remove_date(data):
    new_data = data.copy()
    new_data = new_data.drop('date', 1)
    return new_data


dataset = pd.read_csv("kc_house_data.csv")

print (dataset.columns)

# Pre-processing
dataset = missing_columns(dataset)
dataset = out_of_range_values(dataset)
dataset = remove_id_column(dataset)
dataset = add_interceptor_feature(dataset)
dataset = convert_categorical_features(dataset)
dataset = convert_date(dataset)
dataset = remove_langtitude_longtitude(dataset)
#dataset = remove_date(dataset)

y = dataset["price"]
X = dataset.drop('price', 1).copy()

test_mse = [None] * 100
train_mse = [None] * 100

for x in range(1, 100):
    X_train, y_train, X_test, y_test = split_train_test(X, y, x)
    X_train = X_train.as_matrix().astype(np.float64)
    pinverse = np.linalg.pinv(X_train)
    w_hat = pinverse.dot(y_train)

    # Predict house prices on test set.
    y_predicted_test = X_test.dot(w_hat)

    test_mse[x] = (((y_test - y_predicted_test) ** 2).mean())
    # Predict house prices on train set.
    y_predicted_train = X_train.dot(w_hat)

    train_mse[x] = (((y_train - y_predicted_train) ** 2).mean())
    print("Test MSE :" + str(test_mse[x]))
    print("Train MSE : " + str(train_mse[x]))

plt.figure()
plt.plot(train_mse, label='Training set MSE')
plt.title('Training and Test MSE VS Training Set Sample Percentage')
plt.ylabel('MSE')
plt.xlabel('Percentage of Samples in Training Set')
# plt.figure()
# plt.title('Test set MSE VS Percentage of samples in Training Set')
# plt.ylabel('MSE')
# plt.xlabel('Percentage of Samples in Training Set')

plt.plot(test_mse, label='Test set MSE')
plt.legend()
#plt.savefig('Training and Test MSE VS Percentage of samples in Training Set -without date feature')
plt.show()
