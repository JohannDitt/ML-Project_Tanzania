import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR

from catboost import CatBoostRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from lightgbm import LGBMRegressor

import GetDataset

#Get data
X_features, y_target = GetDataset.get_data()

#define hyperparameters for regressors (optimized in RegressionTree.py, ExtraTreeRegression.py and ForestDecisionTree.py)
param_dec_tree = {'max_depth': 46, 'max_features': 51, 'min_samples_leaf': 0.1, 'min_samples_split': 0.2}
param_ext_tree = {'max_depth': 12, 'max_features': 39, 'min_samples_leaf': 0.1, 'min_samples_split': 0.3}
param_rand_forest = {'max_depth': 10, 'max_features': 10, 'n_estimators': 200}

#Define Regressor for StackingRegressor model
#Hyperparameters had to be defined here again since <Regressor>.set_params didn't work
dec_tree = DecisionTreeRegressor(max_depth=46, max_features=51, min_samples_leaf=0.1, min_samples_split=0.2)
ext_tree = ExtraTreeRegressor(max_depth=12, max_features=39, min_samples_leaf=0.1, min_samples_split=0.3)
rand_forest = RandomForestRegressor(max_depth=10, max_features=10, n_estimators=200)
ext_forest = ExtraTreesRegressor(max_depth=10, max_features=10, n_estimators=200)

#split data
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.25, random_state=42)

#Create list of estimators for stacking regressor
estimators_1 = [
    ("dec_tree", dec_tree),
    ("ext_tree", ext_tree),
    ("rand_forest", rand_forest),
    ("ext_forest", ext_forest),
    ("knn", KNeighborsRegressor())
]

#Create and train stacking regressor
stacking_model = StackingRegressor(estimators_1, final_estimator=Ridge(alpha=0.5), cv=5, n_jobs=2, verbose=5)
stacking_model.fit(X_train, y_train)

#get Predictions
y_pred_test = stacking_model.predict(X_test)
y_pred_train = stacking_model.predict(X_train)

#print metrics
print(f"RMSE test: {mean_squared_error(y_test, y_pred_test)}")
print(f"R2 score test: {r2_score(y_test, y_pred_test)}")

print(f"RMSE train: {mean_squared_error(y_train, y_pred_train)}")
print(f"R2 score train: {r2_score(y_train, y_pred_train)}")

#best result: r2 = 0.451


import matplotlib.pyplot as plt
y_res = (y_test - y_pred_test)/(10**7)
y_pred_test =y_pred_test/(10**7)
y_test = y_test/(10**7)
plt.figure(figsize=(20,6))
plt.subplot(121)

plt.scatter(y_test, y_pred_test)
plt.xlabel("observed costs [10^7 TZS]")
plt.ylabel("predicted costs [10^7 TZS]")
plt.subplot(122)

plt.scatter(y_test, y_res)
plt.xlabel("observed costs [10^7 TZS]")
plt.ylabel("residual [10^7 TZS]")
plt.show()
#Scale numerical values
