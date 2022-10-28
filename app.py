from flask import Flask
app = Flask(__name__)
import os

@app.route('/')
def hello_world():
    

    return 'Hello Sammy!'

# @app.route("/model",methods=['POST'])
# def predict():
#     return {"answer":"Hello"}