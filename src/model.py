import pickle
import os

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    return model, vectorizer
