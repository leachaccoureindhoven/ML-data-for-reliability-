"""@author:  Léa Chaccour & Ben Hasenson"""

from enum import Enum

from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class SplitMode(Enum):
    RANDOM = "random"
    GROUP = "group"


from sklearn.model_selection import StratifiedShuffleSplit

def random_split(data, features, target='Ith', stratify_column='Ith0_flag'):
    X = data[features]
    y = data[target]
    stratify_labels = data[stratify_column]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, stratify_labels))


    



def group_split(data, features, target):
    data["laser_id"] = (
        data["location_wafer"].astype(str) + "_" + data["ParameterID"].astype(str)
    )
    unique_lasers = data["laser_id"].unique()
    train_ids, test_ids = tts(unique_lasers, test_size=0.2, random_state=42)
    train_data = data[data["laser_id"].isin(train_ids)]
    test_data = data[data["laser_id"].isin(test_ids)]
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    return X_train, X_test, y_train, y_test




