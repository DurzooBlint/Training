# Import libraries
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Read in data from CSV as a Pandas dataframe
df = pd.read_csv('Melbourne_housing_FULL.csv')

# Data scrubbing #1: remove unused columns
del df['Address']
del df['Method']
del df['SellerG']
del df['Date']
del df['Postcode']
del df['Lattitude']
del df['Longtitude']
del df['Regionname']
del df['Propertycount']

# Data scrubbing #2: remove rows with missing values
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)

# Data scrubbing #3: Convert non-numeric data to numeric values
df = pd.get_dummies(df, columns= ['Suburb', 'CouncilArea', 'Type'])

# Data scrubbing #4: Split data into X (variables) and y (values)
X = df.drop('Price', axis=1)
y = df['Price']

# Data scrubbing #5: Split data into training and test segments
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# Preview the head of dataframe
print(df.head())

# Select algorithm (gradient boosting regressor) and prepare the model

model = ensemble.GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=5,
    max_features=0.6,
    loss='huber'
)

hyperparameters = {
    'n_estimators': [200, 400],
    'max_depth': [4, 8],
    'min_samples_split': [3, 5],
    'min_samples_leaf': [4, 6],
    'max_features': [0.6, 0.9],
    'loss': ['ls', 'lad', 'huber']
}

# Define grid search, Run with 8 CPUs in parallel.
#grid = GridSearchCV(model, hyperparameters, n_jobs=8)
#grid.fit(X_train, y_train)
model.fit(X_train, y_train)

# Return optimal hyperparameters
#print(grid.best_params_)

# Evaluate the results
#mae_train = mean_absolute_error(y_train, grid.predict(X_train))
mae_train = mean_absolute_error(y_train, model.predict(X_train))
print("Training set Mean Absolute Error: %.2f" % mae_train)

#mae_test = mean_absolute_error(y_test, grid.predict(X_test))
mae_test = mean_absolute_error(y_test, model.predict(X_test))
print("Test set Mean Absolute Error: %.2f" %mae_test)

###################################################### Plotting ######################################################
