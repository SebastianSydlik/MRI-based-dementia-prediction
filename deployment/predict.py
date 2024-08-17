import pickle
from pandas import DataFrame
from numpy import log1p 
from flask import Flask, request, jsonify

with open('pickle/model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)

def predict(X):
    preds = model.predict(X)
    return preds

def prepare_pd(patient):
    return DataFrame.from_dict([patient])

def pred_binary(pred):
    dementia_threshold = log1p(1.5 / 26)
    pred = (pred > dementia_threshold).astype(int)
    return pred

app = Flask('MRI-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    patient = request.get_json()
    
    X_patient = prepare_pd(patient)
    pred = predict(X_patient)
    pred = pred_binary(pred)

    result = {
        'dementia_diagnosis': int(pred[0])
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)