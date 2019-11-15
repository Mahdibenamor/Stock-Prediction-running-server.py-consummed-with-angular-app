import numpy as np
from tensorflow.keras.models import load_model
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify, render_template
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import pickle

from tensorflow.keras.models import Sequential
app = Flask(__name__,template_folder="templates")


#where your function is stored
path="C:/Users/Mahdi Ben Amor/Desktop/stock-prediction/Stock-prediction-model-deploy/scaler.pkl"
file=open(path, 'rb') 
scaler = pickle.load(file)
file.close()

#where your model is stored
model=load_model('C:/Users/Mahdi Ben Amor/Desktop/stock-prediction/Stock-prediction-model-deploy/model.h5')


@app.route('/results',methods=['POST'])
def results():
    X_test = [] 
    data = request.get_json(force=True)
    X_test=data["inputs"]
    X_test = scaler.transform([X_test])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (1,60,1))

    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
  
    return jsonify(float(closing_price))

if __name__ == "__main__":
    app.run(debug=True)