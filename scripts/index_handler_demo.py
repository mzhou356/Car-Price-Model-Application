"""
1. This script reads input from user as a json object.
2. Processes the data into feature format for prediction.
"""
import json
from utils import user_input_process, combine_score 

user_input = json.load(open("user_input.json"))

# load in maps for feature processing 

tree_feature_input, nn_feature_input = user_input_process(user_input)

score = combine_score(tree_feature_input, nn_feature_input)[0]

print(score)
