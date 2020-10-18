"""
This module contains helper functions for data_process.py.
"""
import pickle as pkl
from collections import OrderedDict
import joblib
import tensorflow as tf
import pandas as pd

tfk = tf.keras

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
YEAR_MAP = load_dict("../saved_dict_mappings/year_map.pkl")
MILEAGE_MAP = load_dict("../saved_dict_mappings/mil_dict.pkl")
MPG_MAP = load_dict("../saved_dict_mappings/mpg_dict.pkl")
ENGINE_MAP = load_dict("../saved_dict_mappings/engine_dict.pkl")
BINNING_MAPS = [YEAR_MAP, MILEAGE_MAP, MPG_MAP, ENGINE_MAP]
TREE_COLS = load_dict("../saved_dict_mappings/tree_feature_cols.pkl")
NN_COLS = load_dict("../saved_dict_mappings/nn_feature_cols.pkl")
CATE_MAP = load_dict("../saved_dict_mappings/cate_map.pkl")
EMBED_COLS = ["model", "transmission", "brand", "fuelType", "year",
              "mil_cat", "binned_mpg", "engine_binned"]
RF_MDL = joblib.load("../saved_models/rf_price_regressor.sav")
NN_MDL = tfk.models.load_model("../saved_models/final_embed_mdl.h5",
                               custom_objects={"leaky_relu":tf.nn.leaky_relu})

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

def nn_input_process(user_input, user_input_processed_1):
    """
    This function takes modified user_input and
    user_input_processed_1 dictionary to make neural network
    features.

    arg:
    user_input: a json object, modified by input_process.
    user_input_processed_1: output from input_process.

    return:
    neural network feature numpy array for prediction.
    """
    combined_inputs = {**user_input, **user_input_processed_1}
    data = pd.DataFrame(combined_inputs, index=[0])[NN_COLS]
    return data

def user_input_process(user_input):
    """
    This function takes raw user_input and returns tree and
    nn_features for prediction.

    arg:
    user_input: a json object.

    return:
    a python tuple: tree_features as numpy array, nn_features as dataframe.
    """
    user_input_process_1 = input_process(user_input)
    tree_features = tree_input_process(user_input, user_input_process_1)
    nn_features = nn_input_process(user_input, user_input_process_1)
    return tree_features, nn_features

def data_convert(inputs):
    """
    This function takes neural network feature and makes it into categorical
    embedding input list.

    arg:
    inputs: output from user_input_process for nn_features.

    return:
    a list of categorical embedding feature list for prediction.
    """
    cate_feature_list = []
    for col in EMBED_COLS:
        cate_feature_list.append(inputs[col].map(CATE_MAP[col]).fillna(0).values)
    return cate_feature_list

def combine_score(tree_features, nn_features):
    """
    This function takes tree_feature and nn_features and perform
    combined predicted car price.

    arg:
    tree_features: output of user_input_process, a numpy array.
    nn_features: output of user_input_process, a pandas dataframe.

    returns:
    combined predicted car price.
    """
    nn_input_feature = data_convert(nn_features)
    pred_tree = RF_MDL.predict(tree_features)
    pred_nn = NN_MDL.predict(nn_input_feature, batch_size=512)
    combined = pred_nn.flatten()*0.5 + pred_tree*(1-0.5)
    return combined
