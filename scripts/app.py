import flask
from flask import request, jsonify
from utils import user_input_process, combine_score

app = flask.Flask(__name__)

@app.route("/carPrice", methods=["POST"])
def predict():
        content = request.get_json()
        tree_feature, nn_feature = user_input_process(content)
        score = combine_score(tree_feature, nn_feature)[0]
        return jsonify(score)
	
if __name__ == "__main__":
    app.run(host="0.0.0.0", threaded=True, port=8080, debug=True, use_reloader=True)
