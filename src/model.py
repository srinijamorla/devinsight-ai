import pickle
import os

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer
