#!usr/bin/python
"""
1. This script reads a single input from user as a json object.
2. Processes the data into feature format for prediction.
3. Outputs a score as dictionary/json format.
"""
import json
from utils import user_input_process, score

# load in a list of test user inputs from training data.

def lambda_handler(event,context):
	USER_INPUT = json.load(open("test_input.json"))
	TREE_FEATURE = user_input_process(USER_INPUT)
	SCORE = score(TREE_FEATURE)[0]
	
	return {
		"predicted_price": SCORE,
		"statusCode":200,
		"body":json.dumps("Predicted the car price!")
	}