#!usr/bin/python
"""
1. This script reads a single input from user as a json object.
2. Processes the data into feature format for prediction.
3. Outputs a score as dictionary/json format.
"""
import json
from utils import user_input_process, combine_score

# load in a list of test user inputs from training data.
USER_INPUT = json.load(open("test_input.json"))

TREE_FEATURE, NN_FEATURE = user_input_process(USER_INPUT)
SCORE = combine_score(TREE_FEATURE, NN_FEATURE)[0]

OUTPUT = {"pred_price":SCORE}

print(OUTPUT, flush=True)
