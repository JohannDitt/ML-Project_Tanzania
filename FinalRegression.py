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
#from FitPredict import *

#Get data
X_features, y_target = GetDataset.get_data()

#split data
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.25, random_state=42)

#Define regressors
dec_tree = DecisionTreeRegressor(max_depth=46, max_features=51, min_samples_leaf=0.1, min_samples_split=0.2)
ext_tree = ExtraTreeRegressor(max_depth=12, max_features=39, min_samples_leaf=0.1, min_samples_split=0.3)
rand_forest = RandomForestRegressor(max_depth=10, max_features=10, n_estimators=200)
ext_forest = ExtraTreesRegressor(max_depth=10, max_features=10, n_estimators=200)

#create list of regressors for stacking regressor
estimators_1 = [
    ("dec_tree", dec_tree),
    ("ext_tree", ext_tree),
    ("rand_forest", rand_forest),
    ("ext_forest", ext_forest),
    ("knn", KNeighborsRegressor())
]

#split train set half-half
X_train1, X_train2, y_train1, y_train2 = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

#Create first stacking regressor and train it on half of training set
stacking_model = StackingRegressor(estimators_1, final_estimator=Ridge(alpha=0.5), cv=5, n_jobs=2, verbose=5)
stacking_model.fit(X_train1, y_train1)

#get prediction from stacking regressor for the second half of trainng set
y_pred_train2 = stacking_model.predict(X_train2)

#create linear regression model
lin_reg = LinearRegression()

#use prediction as feature for linear regressor and train it
lin_reg.fit(y_pred_train2.reshape(-1,1), y_train2)

#get overall predcition for whole training set
y_pred_train = stacking_model.predict(X_train)
y_pred_train = lin_reg.predict(y_pred_train.reshape(-1,1))

#get overall prediction for test set
y_pred_test1 = stacking_model.predict(X_test)
y_pred_test2 = lin_reg.predict(y_pred_test1.reshape(-1,1))

#print metrics
print(f"RMSE test: {mean_squared_error(y_test, y_pred_test2)}")
print(f"R2 score test: {r2_score(y_test, y_pred_test2)}")

print(f"RMSE train: {mean_squared_error(y_train, y_pred_train)}")
print(f"R2 score train: {r2_score(y_train, y_pred_train)}")

#print parameters of linear regression
print(lin_reg.coef_, lin_reg.intercept_)

#plot results (predicted vs observed, residual vs. observed)
import matplotlib.pyplot as plt
y_test = y_test/(10**7)
y_pred_test2 = y_pred_test2/(10**7)
y_res = y_test - y_pred_test2

plt.figure(figsize=(20,6))
plt.subplot(121)
plt.scatter(y_test, y_pred_test2)
plt.xlabel("observed costs [10^7 ZS]")
plt.ylabel("predicted costs [10^7 TZS]")
plt.subplot(122)
plt.scatter(y_test, y_res)
plt.xlabel("observed costs [10^7 TZS]")
plt.ylabel("residual [10^7 TZS]")
plt.show()