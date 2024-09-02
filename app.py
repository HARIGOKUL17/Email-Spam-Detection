import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("Spam_Model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    message = request.form['message']
    data = [message]
    cv = CountVectorizer()
    vect= cv.transform(data).toarray()
    my_prediction = model.predict(data)

    return render_template('index.html', prediction=my_prediction)

if __name__ == "__main__":
    flask_app.run(debug=True)