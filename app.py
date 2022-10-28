
from flask import Flask, request
import json
import base64
from model import model_fit

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello Sammy!'

@app.route("/model",methods=['POST'])
def predict():
        data = request.json
        imgdata = base64.b64decode(data["image"].split(',')[1])
        filename = 'image.jpg'
        with open(filename, 'wb') as f:
                f.write(imgdata)
        result = model_fit(data['question'], 'image.jpg')
        answer = "Sorry, I can't answer your question. Please ask another question."
        if result == 1:
                answer = "Ok, I can answer your question."
        json_object = json.dumps({"message": answer}) 
        return json_object