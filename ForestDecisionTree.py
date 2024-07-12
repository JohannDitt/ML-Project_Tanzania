import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

import matplotlib.pyplot as plt
import seaborn as sns

import GetDataset

#Get data using GetDataset
X_features, y_target = GetDataset.get_data()

#split in train and test set
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.25, random_state=42)

#Hyperparameters from best DecisionTreeRegressor (optimized in RegressionTree.py)
max_depth = 46
max_features = 51
min_samples_leaf = 0.1
min_samples_split = 0.2
criterion = "squared_error"

#Train random forest with parameters of the best decision tree (RegressionTree.py)
rand_forest = RandomForestRegressor(criterion=criterion, max_depth=max_depth, max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
rand_forest.fit(X_train, y_train)

#get predictions
y_pred = rand_forest.predict(X_test)
y_pred_train = rand_forest.predict(X_train)

#print metrics
print(f"RMSE test: {mean_squared_error(y_test, y_pred, squared=False)}")
print(f"R2 score test: {r2_score(y_test, y_pred)}")

print(f"RMSE train: {mean_squared_error(y_train, y_pred_train, squared=False)}")
print(f"R2 score train: {r2_score(y_train, y_pred_train)}")

#Define grid for Grid search
param_grid = {
    "n_estimators": [50, 100, 200, 500], #number of trees 
    "max_depth": [10, 20, 30, 40], #maximum depth of trees
    "max_features": [10, 20, 30, 40] # maximum number of features used in a single tree
}

#Search for best Random forest
rand_forest1 = RandomForestRegressor()
grid_search = GridSearchCV(rand_forest1, param_grid, verbose=0, n_jobs=2)
grid_search.fit(X_train, y_train)

#Get best Random Forest model
print(grid_search.best_params_)

rand_forest_best = grid_search.best_estimator_

#get predictions
y_pred = rand_forest_best.predict(X_test)
y_pred_train = rand_forest_best.predict(X_train)

#print metrics
print(f"RMSE test: {mean_squared_error(y_test, y_pred, squared=False)}")
print(f"R2 score test: {r2_score(y_test, y_pred)}")

print(f"RMSE train: {mean_squared_error(y_train, y_pred_train, squared=False)}")
print(f"R2 score train: {r2_score(y_train, y_pred_train)}")

#plot the residual vs. observed values 
y_residuals = y_test - y_pred
plt.scatter(y_test, y_residuals)
plt.xlabel("observed")
plt.ylabel("residual")
plt.show()


#Result of best random forest model: r2 = 0.412


#RandomForest of ExtraTreeRegressors
ext_trees = ExtraTreesRegressor()

#optimize Hyperparameters for ExtraTreesRegressor
grid_search1 = GridSearchCV(ext_trees, param_grid, verbose=0, n_jobs=2)
grid_search1.fit(X_train, y_train)

#Get best Extra Trees model
print(grid_search1.best_params_)

ext_forest_best = grid_search1.best_estimator_

#get prediction of best model
y_pred = ext_forest_best.predict(X_test)
y_pred_train = ext_forest_best.predict(X_train)

#print metrics
print(f"RMSE test Ext: {mean_squared_error(y_test, y_pred, squared=False)}")
print(f"R2 score test Ext: {r2_score(y_test, y_pred)}")

print(f"RMSE train Ext: {mean_squared_error(y_train, y_pred_train, squared=False)}")
print(f"R2 score train Ext: {r2_score(y_train, y_pred_train)}")

#Result of best ExtraTrees Regressorr2= 0.399