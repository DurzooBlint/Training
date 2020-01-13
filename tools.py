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
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

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
    # lin_reg.fit(samples, labels)
    tree_reg = DecisionTreeRegressor()
    # tree_reg.fit(samples, labels)
    random_forest_reg = RandomForestRegressor()
    # random_forest_reg.fit(samples, labels)
    svm_linear_reg = SVR()
    # svm_linear_reg2 = SVR(kernel='rbf', C=0.5)
    # svm_linear_reg3 = SVR(kernel='linear', gamma='auto')
    # svm_linear_reg.fit(samples, labels)
    # svm_linear_reg2.fit(samples, labels)
    # svm_linear_reg3.fit(samples, labels)
    # cross validate
    # lin_reg_scores = cross_val_score(lin_reg, samples, labels,
    #                                  scoring="neg_mean_squared_error", cv=10)
    # tree_reg_scores = cross_val_score(tree_reg, samples, labels,
    #                                   scoring="neg_mean_squared_error", cv=10)
    # random_forest_scores = cross_val_score(random_forest_reg, samples, labels,
    #                                        scoring="neg_mean_squared_error", cv=10)
    # svm_linear_scores = cross_val_score(svm_linear_reg, samples, labels,
    #                                     scoring="neg_mean_squared_error", cv=10)
    # svm_linear_scores2 = cross_val_score(svm_linear_reg2, samples, labels,
    #                                     scoring="neg_mean_squared_error", cv=10)
    # svm_linear_scores3 = cross_val_score(svm_linear_reg3, samples, labels,
    #                                     scoring="neg_mean_squared_error", cv=10)
    # lin_rmse_scores = (np.sqrt(-lin_reg_scores)).mean()
    # tree_rmse_scores = (np.sqrt(-tree_reg_scores)).mean()
    # forest_rmse_scores = (np.sqrt(-random_forest_scores)).mean()
    # svm_rmse_scores = (np.sqrt(-svm_linear_scores)).mean()
    # svm_rmse_scores2 = (np.sqrt(-svm_linear_scores2)).mean()
    # svm_rmse_scores3 = (np.sqrt(-svm_linear_scores3)).mean()
    #
    # print('Linear Regression RMSE:', lin_rmse_scores)
    # print('Decision Tree Regression RMSE:', tree_rmse_scores)
    # print('Random Forst Regression RMSE:', forest_rmse_scores)
    # print('SVM linear Regression RMSE:', svm_rmse_scores)
    # print('SVM linear Regression2 RMSE:', svm_rmse_scores2)
    # print('SVM linear Regression3 RMSE:', svm_rmse_scores3)

    # perform GridSearch for all models and print out score

    param_grid_linear = param_grid = {'fit_intercept': ['False', 'True'], 'normalize': ['False', 'True'],
                                      'n_jobs': [-1]}
    param_grid_decision_tree = {'splitter': ['best', 'random'], 'min_samples_split': [2, 4], 'min_samples_leaf': [1, 2, 4],
                               'max_features': ['auto', 2, 4, 6, 8]}
    param_grid_random_forest = {'n_estimators': [100, 200, 300], 'min_samples_split': [2, 4],
                               'max_features': ['auto', 2, 4, 6, 8], 'min_samples_leaf': [1, 2, 4], 'n_jobs': [-1]}
    param_grid_svm = {'C': [0.001, 0.1, 10, 100, 10e5], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf', 'poly']}

    grid_search = GridSearchCV(lin_reg, param_grid_linear, cv=10)
    grid_search.fit(samples, labels)
    print('Linear Regression score = %3.2f' %(grid_search.score(samples,labels)))

    grid_search = GridSearchCV(svm_linear_reg, param_grid_svm, cv=10)
    grid_search.fit(samples, labels)
    print('SVM score = %3.2f' %(grid_search.score(samples,labels)))

    grid_search = GridSearchCV(tree_reg, param_grid_decision_tree, cv=10)
    grid_search.fit(samples, labels)
    print('Decision Tree Regression score = %3.2f' %(grid_search.score(samples,labels)))

    grid_search = GridSearchCV(random_forest_reg, param_grid_random_forest, cv=10)
    grid_search.fit(samples, labels)
    print('Random Forest Regression score = %3.2f' %(grid_search.score(samples,labels)))

