import os
import joblib
from flask import Flask, request, jsonify, render_template

# Initialize the Flask app instance
app = Flask(__name__)

# Load our pre-trained model once when the web server boots up
MODEL_PATH = "iris_model.pkl"
model = joblib.load(MODEL_PATH)

# Human-readable mapping for the Iris classification indices
FLOWER_MAP = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

@app.route("/", methods=["GET"])
def home():
    """Serve the web UI home page template."""
    return render_template("index.html"), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Accepts feature JSON, processes it, and returns the predicted flower class."""
    try:
        # Parse the incoming JSON request payload
        data = request.get_json()
        
        # Structure the features into the matrix shape expected by scikit-learn
        features = [[
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
