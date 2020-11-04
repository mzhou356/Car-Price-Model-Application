import flask
from flask import request, jsonify
from waitress import serve
from utils import user_input_process, combine_score

app = flask.Flask(__name__)

@app.after_request
def after_request(response):
	response.headers.add("Access-Control-Allow-Origin", "*")
	response.headers.add("Access-Control-Allow-Headers", "Content-Type")
	response.headers.add("Access-Control-Allow-Methods", "POST,OPTIONS")
	return response


@app.route("/carPrice", methods=["POST"])
def predict():
    content = request.get_json()
#     print(content)
    if content is None:
    	return jsonify({"error": "invalid input"}), 400
    tree_feature, nn_feature = user_input_process(content)
    score = round(combine_score(tree_feature, nn_feature)[0],2)
    return jsonify({"pred_price":score}), 200
	
if __name__ == "__main__":
 #    app.run(host="0.0.0.0", threaded=True, port=8081, debug=True, use_reloader=True)
    serve(app, host="0.0.0.0", port=8081)
