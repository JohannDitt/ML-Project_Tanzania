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

def fit_models(X, y, *models):
    """Function to train a stacked model with more than one stage. Output of one stage is used as input for the next stage.

    Args:
        X (2D array): Dataset containing the features
        y (1D array): Dataset containing the targets
        *models (Regressors): list of models for stacking. !!!Models must be in the right order!!!

    Returns:
        1D array: overall prediction for the trainingset
    """
    #Create X-train dataset as copy of X
    X_train = X
    
    #loop over all models
    for model in models:
        try: #train with training set and predict training set if trainign set has correct dimensions 
            model.fit(X_train, y)
            X_train = model.predict(X_train)
        except ValueError: #.fit and .predict need 2D array as input 
            X_train = X_train.reshape(-1, 1) #bring dataset in the right dimensions
            model.fit(X_train, y) #train the model
            X_train = model.predict(X_train) #predict values for traing data
            
    #return overall prediction for training set
    return X_train
    
    
    

def get_prediction(X, *models):
    """Function to train a stacked model with more than one stage. Output of one stage is used as input for the next stage.

    Args:
        X (2D array): SDataset containign features
        *models (Regressors): list of models for stacking. !!!Models must be in the right order!!!

    Returns:
        1D array: overall prediction 
    """
    #create array for predictions
    y_pred = X
    #loop over all models
    for model in models:
        try:#predict values if imput is in correct dimension
            y_pred = model.predict(y_pred)
        except ValueError:#bring input in right dimension if not done yet, and predict values
            y_pred = y_pred.reshape(-1,1)
            y_pred = model.predict(y_pred)
        
    #return overall predictions
    return y_pred