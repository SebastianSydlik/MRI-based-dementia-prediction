import pickle

with open('models/xgb.bin', 'rb') as f_in:
    (model) = pickle.load(f_in)

def predict(X):
    preds = model.predict(X)
    return preds
