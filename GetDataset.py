import pandas as pd
import numpy as np

def get_data():
    #import dataset
    data = pd.read_csv("data/data_cleaned.csv")

    #Drop unneccessary columns
    data = data.drop(["ID", "country", "night_total", "group_size"], axis=1)

    #Create lists for numerical and categorical features
    num_features = ["tour_arrangement", "package_transport_int", "package_accomodation", "package_food", "package_transport_tz", "package_sightseeing", "package_guided_tour", "package_insurance", "first_trip_tz"]
    cont_num_features = ["total_female", "total_male", "night_mainland", "night_zanzibar"]
    target = "total_cost"
    cat_features = list(data.columns)

    for x in num_features:
        cat_features.remove(x)
    
    for x in cont_num_features:
        cat_features.remove(x)

    cat_features.remove(target)
    cat_features = cat_features[1:]

    #Create Sets for features and targets
    y_target = data[target]

    features = num_features + cont_num_features + cat_features
    X_features = data[features]

    #Get dummies for categories
    X_features = pd.get_dummies(X_features, columns=cat_features, dtype=int)
    
    return X_features, y_target


def get_data_cutted():
    
    x_features, y_target = get_data()
    
    #define cutting bins
    cost_bins = np.arange(0, 10**8+1, 10**7)
    
    labels = cost_bins//(10**7)
    
    y_target = pd.cut(y_target, cost_bins, labels=labels[1:])
    
    return x_features, y_target
    
if __name__ == "__main__":
    
    X, y = get_data()
    
    print(min(y), max(y))
    
    X1, y1 = get_data_cutted()
    
    print(y1.value_counts())