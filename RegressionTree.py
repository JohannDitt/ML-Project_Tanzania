import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import GetDataset

X_features, y_target = GetDataset.get_data()

print(f"Number of features: {X_features.shape[1]}")

#Split data set
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.25, random_state=42)

#Create regression tree
reg_tree = DecisionTreeRegressor()

reg_tree.fit(X_train, y_train)

#get predictions
y_pred = reg_tree.predict(X_test)

print(f"RMSE test: {mean_squared_error(y_test, y_pred, squared=False)}")
print(f"R2 score test: {r2_score(y_test, y_pred)}")

y_pred_train = reg_tree.predict(X_train)

print(f"RMSE train: {mean_squared_error(y_train, y_pred_train, squared=False)}")
print(f"R2 score train: {r2_score(y_train, y_pred_train)}")

print(f"Hyperparameters: {reg_tree.get_depth()}")

#This tree is not good and overfitted

#Use A grid search cv
param_grid = {
    "max_depth": np.arange(2, 53, 1),
    "max_features": np.arange(1, 53, 1),
    "min_samples_leaf": np.arange(0, 0.6, 0.1),
    "criterion" : ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    "min_samples_split": np.arange(0.05, 1.0, 0.05)
}

grid_search = GridSearchCV(reg_tree, param_grid, cv=3, verbose=10)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

reg_tree_best = grid_search.best_estimator_

y_pred_best = reg_tree_best.predict(X_test)

print(f"RMSE best test: {mean_squared_error(y_test, y_pred_best, squared=False)}")
print(f"R2 score best test: {r2_score(y_test, y_pred_best)}")

