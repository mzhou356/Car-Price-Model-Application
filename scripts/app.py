import flask
from flask import request, jsonify

app = flask.Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return """<h1>Distant Reading Archive</h1>
              <p>A prototype API for distant reading of science fiction novels.</p>"""
              
if __name__ == "__main__":
    app.run(threaded=True, port=5000, debug=True, use_reloader=True)
