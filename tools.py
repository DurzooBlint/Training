import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np

# Constants
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


# Download compressed csv file and extract
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    np.random.seed(77)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def display_scores(scores, label):
    print("\n###", label, "###")
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def get_best_model(samples, labels, hypersearch=False):
    # train different kind of models
    lin_reg = LinearRegression()
    lin_reg.fit(samples, labels)
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(samples, labels)
    random_forest_reg = RandomForestRegressor()
    random_forest_reg.fit(samples, labels)

    # cross validate
    lin_reg_scores = cross_val_score(lin_reg, samples, labels,
                                     scoring="neg_mean_squared_error", cv=10)
    tree_reg_scores = cross_val_score(tree_reg, samples, labels,
                                      scoring="neg_mean_squared_error", cv=10)
    random_forest_scores = cross_val_score(random_forest_reg, samples, labels,
                                           scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = (np.sqrt(-lin_reg_scores)).mean()
    tree_rmse_scores = (np.sqrt(-tree_reg_scores)).mean()
    forest_rmse_scores = (np.sqrt(-random_forest_scores)).mean()

    print(lin_rmse_scores)
    print(tree_rmse_scores)
    print(forest_rmse_scores)
