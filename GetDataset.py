import pandas as pd
import numpy as np

def get_data():
    """Function to create datasets for feutures and targets based on cleaned data from Tanzania Tourism Predicition dataset

    Returns:
        DataFrame/2D array (X_feature): Dataset of all features
        Series/1D array (y_target); Dataset of target values (total_cost in this case)
    """
    
    #import dataset
    data = pd.read_csv("data/data_cleaned.csv")

    #Drop unneccessary columns
    data = data.drop(["ID", "night_total", "group_size"], axis=1)

    #Create lists for numerical and categorical features
    num_features = ["tour_arrangement", "package_transport_int", "package_accomodation", "package_food", "package_transport_tz", "package_sightseeing", "package_guided_tour", "package_insurance", "first_trip_tz"]
    cont_num_features = ["total_female", "total_male", "night_mainland", "night_zanzibar"]
    target = "total_cost"
    #iniate all columns as categorical
    cat_features = list(data.columns)

    #remove binaric numerical features
    for x in num_features:
        cat_features.remove(x)
    
    #remove continoous numerical features
    for x in cont_num_features:
        cat_features.remove(x)

    #remove target column name
    cat_features.remove(target)
    
    #remove "Unnamed: 0"
    cat_features = cat_features[1:]

    #Create target set
    y_target = data[target]

    #Create feature set
    features = num_features + cont_num_features + cat_features
    X_features = data[features]

    #Get dummies for categories and create new dataframe
    X_features = pd.get_dummies(X_features, columns=cat_features, dtype=int)
    
    #return Datasets for features and target
    return X_features, y_target


def get_data_cutted():
    """This function changes the target values to categorical values. The output is used for classification.

    Returns:
        X_features (DataFrame/2D array): Dataset of all features. Same as get_data()-function
        y_target (Series/1D array): Dataset of target as classes
    """
    
    #Get datasets from function above
    x_features, y_target = get_data()
    
    #define cutting bins
    cost_bins = np.arange(0, 10**8+1, 2*(10**7))
    
    #Define labels used as classes
    labels = cost_bins//(10**7)
    
    #cut target values into bins and createnew dataset with lables 
    y_target = pd.cut(y_target, cost_bins, labels=labels[1:])
    
    #return datasets
    return x_features, y_target



#code below was used to test functions above
if __name__ == "__main__":
    
    X, y = get_data()
    
    print(min(y), max(y))
    
    X1, y1 = get_data_cutted()
    
    print(y1.value_counts())