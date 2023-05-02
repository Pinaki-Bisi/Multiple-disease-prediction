import joblib
from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

model= pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))

app = Flask(__name__,template_folder="templates",static_folder="static")

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/diabetics_index.html")
def diabetics_index():
    return render_template("diabetics_index.html")

@app.route('/predictdia', methods=['POST'])
def predictdia():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    output = model2.predict(features)
    

    if output == 1:
        res_val = "a high risk of dia Disease"
    else:
        res_val = "a low risk of dia Disease"


    return render_template('diabetics_index.html', prediction_text='Patient has {}'.format(res_val))

@app.route("/heart_index.html")
def heart_index():
    return render_template("heart_index.html")

@app.route('/predictheart', methods=['POST'])
def predictheart():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    output = model.predict(features)
    

    if output == 1:
        res_val = "a high risk of Heart Disease"
    else:
        res_val = "a low risk of Heart Disease"

    return render_template('heart_index.html', prediction_text='Patient has {}'.format(res_val))

@app.route("/kidney_index.html")
def kidney_index():
    return render_template("kidney_index.html")

@app.route("/predictkidney",  methods=['GET', 'POST'])
def predictkidney():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    output = model1.predict(features)
    

    if output == 1:
        res_val = "a high risk of kidney Disease"
    else:
        res_val = "a low risk of kidney Disease"

    return render_template('kidney_index.html', prediction_text='Patient has {}'.format(res_val))

@app.route("/doc.html")
def doc_index():
    return render_template("doc.html")

@app.route("/gallery.html")
def gallery_index():
    return render_template("gallery.html")

@app.route("/index.html")
def index():
    return render_template("index.html")
@app.route("/review.html")
def review():
    return render_template("review.html")


if __name__ == "__main__":
    app.run(debug=True)