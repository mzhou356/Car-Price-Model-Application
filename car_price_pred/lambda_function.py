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
	user_input = json.loads(event["body"])
	try:	
		TREE_FEATURE = user_input_process(user_input)
		SCORE = score(TREE_FEATURE)[0]
		return {
		"statusCode":200,
		"headers":{"Content-Type": "application/json"},
		"body":json.dumps({"predicted_price":SCORE})
		}
	except:
		return {
		"statusCode":500,
		"headers":{"Content-Type": "application/json"},
		"body": ""
		}