from flask import Flask, render_template, request
import pandas as pd
from src.model import load_model
from src.preprocess import clean_text

app = Flask(__name__)

# Load the model and vectorizer once
model, vectorizer = load_model()

@app.route('/')
def index():
    df = pd.read_csv('data/commits.csv')
    table_html = df.to_html(classes='table table-striped', index=False)
    return render_template('index.html', table=table_html)

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        df = pd.read_csv(uploaded_file)
        if 'commit_message' not in df.columns:
            return "CSV must have a 'commit_message' column.", 400
        
        # Clean and vectorize
        df['cleaned'] = df['commit_message'].apply(clean_text)
        X = vectorizer.transform(df['cleaned'])
        df['predicted_label'] = model.predict(X)

        table_html = df[['commit_message', 'predicted_label']].to_html(classes='table table-bordered', index=False)
        return render_template('index.html', table=table_html)
    return "No file uploaded.", 400

if __name__ == '__main__':
    app.run(debug=True)
