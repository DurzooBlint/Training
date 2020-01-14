#Author : Marcin Karpik

import tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# 1. Load data #######################################################################
housing = tools.load_housing_data()

# print(housing.info())
# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

# 2. Prepare data #######################################################################
# create categories from income attribute
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
# creation of new, more useful attributes
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"] = housing["population"]/housing["household"]

# split data into train and test sets using new income_cat attribute
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# remove "income_cat" attribute from datasets
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# from now one we work on the copy of train set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# fill missing values for numerical attributes
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

# and for non numerical by creation categories and changing them to numbers
housing_cat = housing[["ocean_proximity"]]
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# create binary attributes from categories
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# custom transformer to add extra attributes:
# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# apply transformers using pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform((housing))

# 3. Visualization of data #######################################################################
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#             s=housing["population"] / 100, label="population", figsize=(10, 7),
#             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
# plt.legend()
# plt.show()

# 4. Train different models and select most accurate #######################################################################

tools.get_best_model(housing_prepared, housing_labels, 'randomized')


# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)
#
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)
#
# forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
#
# # 5. Test model #######################################################################
#
# # 5.1 Measure accuracy with Mean Squared Error
# housing_predictions = lin_reg.predict(housing_prepared)
# housing_predictions_tree = tree_reg.predict(housing_prepared)
# housing_predictions_forest = forest_reg.predict(housing_prepared)
#
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# tree_mse = mean_squared_error(housing_labels, housing_predictions_tree)
# tree_rmse = np.sqrt(tree_mse)
# forest_mse = mean_squared_error(housing_labels, housing_predictions_forest)
# forest_rmse = np.sqrt(forest_mse)
#
# print("Linear Regression RMSE: ", lin_rmse)
# print("Tree Regression RMSE: ", tree_rmse)
# print("Random forest Regression RMSE: ", forest_rmse)
#
# # 5.2 Cross-validation
# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
#                              scoring="neg_mean_squared_error", cv=10)
# tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
#                               scoring="neg_mean_squared_error", cv=10)
# forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
#                               scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
# tree_rmse_scores = np.sqrt(-tree_scores)
# forest_rmse_scores = np.sqrt(-forest_scores)
# tools.display_scores(lin_rmse_scores, "Linear regression")
# tools.display_scores(tree_rmse_scores, "Decision tree regression")
# tools.display_scores(forest_rmse_scores, "Random forest regression")
#
# # 5.3 Looking for best hyperparameters using GridSearch
# param_grid = [
#     # try 12 (3×4) combinations of hyperparameters
#     {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
#     # then try 6 (2×3) combinations with bootstrap set as False
#     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
#   ]
#
# forest_reg = RandomForestRegressor(random_state=42)
# # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
#                            scoring='neg_mean_squared_error',
#                            return_train_score=True)
# grid_search.fit(housing_prepared, housing_labels)
#
# # 5.4 Evaluate on test set
# final_model = grid_search.best_estimator_
#
# X_test = strat_test_set.drop("median_house_value", axis=1)
# y_test = strat_test_set["median_house_value"].copy()
#
# X_test_prepared = full_pipeline.transform(X_test)
# final_predictions = final_model.predict(X_test_prepared)
#
# final_mse = mean_squared_error(y_test, final_predictions)
# final_rmse = np.sqrt(final_mse)
# tools.display_scores(final_rmse, "Random Forest regression using GridSearch")
