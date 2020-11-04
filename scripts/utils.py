"""
This module contains helper functions for data_process.py.
"""
import os
import pickle as pkl
from collections import OrderedDict
import joblib
import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))

# define helper functions.
def load_dict(filename):
    """
    Load in a pickle format file object.

    arg:
    filename: a python string, filepath name.

    return:
    a python object as a dictionary.
    """
    with open(filename, "rb") as handle:
        dict_mapping = pkl.load(handle)
    return dict_mapping

# define constants:
YEAR_MAP = load_dict(os.path.join(PATH, "year_map.pkl"))
MILEAGE_MAP = load_dict(os.path.join(PATH, "mil_dict.pkl"))
MPG_MAP = load_dict(os.path.join(PATH, "mpg_dict.pkl"))
ENGINE_MAP = load_dict(os.path.join(PATH, "engine_dict.pkl"))
BINNING_MAPS = [YEAR_MAP, MILEAGE_MAP, MPG_MAP, ENGINE_MAP]
TREE_COLS = load_dict(os.path.join(PATH, "tree_feature_cols.pkl"))
RF_MDL = joblib.load(os.path.join(PATH, "rf_price_regressor.sav"))

def binning(col, thresholds):
    """
    This function bins numerical features into categories.

    Args:
    col: a numeric value for the feature column
    thresholds: a dictionary for map between threshold and categorical value
    df: a pandas dataframe, the original data.

    return:
    transformed col as a categorical object type.
    """
    for key, value in thresholds.items():
        if key[0] <= col <= key[1]:
            return value
    return None

def input_process(user_input):
    """
    This function takes raw user_input and map numerica data to binned
    values.

    arg:
    user_input: a json object.

    return:
    a python dictionary with binned numerical features.
    """
    cols = ["year", "mileage", "mpg", "engineSize"]
    binned_columns = ["binned_year", "mil_cat", "binned_mpg", "engine_binned"]
    processed = {}
    for i, col in enumerate(cols):
        original_value = user_input[col]
        converted_value = int(binning(original_value, BINNING_MAPS[i]))
        processed[binned_columns[i]] = converted_value
        if col != "year":
            user_input.pop(col)
    return processed

def tree_input_process(user_input, user_input_processed_1):
    """
    This function takes modified user_input and
    user_input_processed_1 dictionary to make random forest
    features.

    arg:
    user_input: a json object, modified by input_process.
    user_input_processed_1: output from input_process.

    return:
    random forest feature numpy array for prediction.
    """
    user_input_tree_dict = OrderedDict.fromkeys(TREE_COLS, 0)
    for key, _ in user_input_tree_dict.items():
        if key in user_input_processed_1:
            user_input_tree_dict[key] = user_input_processed_1[key]
            continue
        for _, val in user_input.items():
            if str(val) in key:
                user_input_tree_dict[key] = 1
    return pd.DataFrame(user_input_tree_dict, index=[0]).values



def user_input_process(user_input):
    """
    This function takes raw user_input and returns tree features.

    arg:
    user_input: a json object.

    return:
    tree_features as numpy array
    """
    user_input_process_1 = input_process(user_input)
    tree_features = tree_input_process(user_input, user_input_process_1)
    return tree_features



def score(tree_features):
    """
    This function takes tree_feature perform predicted car price.

    arg:
    tree_features: output of user_input_process, a numpy array.

    returns:
    predicted car price.
    """
    pred_tree = RF_MDL.predict(tree_features)
    return pred_tree
