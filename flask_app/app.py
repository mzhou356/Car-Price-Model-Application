import flask
from flask import request, jsonify
from waitress import serve
from utils import user_input_process, combine_score

app = flask.Flask(__name__)

@app.route("/carPrice", methods=["POST"])
def predict():
    content = request.get_json()
    if content is None:
    	return jsonify({"error": "invalid input"}), 400
    tree_feature, nn_feature = user_input_process(content)
    score = combine_score(tree_feature, nn_feature)[0]
    return jsonify({"pred_price":score}), 200
	
if __name__ == "__main__":
    app.run(host="0.0.0.0", threaded=True, port=8081, debug=True, use_reloader=True)
    # serve(app, host="0.0.0.0", port=8081)
