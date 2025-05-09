# app.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename

from src.preprocess import clean_text
from src.model import load_model
from src.github_fetch import fetch_commits  # ✅ Import your function

app = Flask(__name__)

current_dir = os.getcwd()
UPLOAD_FOLDER = os.path.join(current_dir, 'src', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model & vectorizer once
model, vectorizer = load_model()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    df = pd.read_excel(filepath)
    if 'message' not in df.columns:
        return jsonify({"error": "Missing 'message' column"}), 400

    df['cleaned'] = df['message'].apply(clean_text)
    X = vectorizer.transform(df['cleaned'])
    df['label'] = model.predict(X)

    output_path = os.path.join(UPLOAD_FOLDER, f"predicted_{filename}")
    df[['message', 'label']].to_excel(output_path, index=False)

    return send_file(output_path, as_attachment=True)


@app.route('/analyze_repo', methods=['POST'])
def analyze_repo():
    repo_url = request.form.get('repo_url')
    if not repo_url:
        return jsonify({"error": "No GitHub repo URL provided"}), 400

    # ✅ Fetch commits using your logic
    df = fetch_commits(repo_url)

    if df.empty:
        return jsonify({"error": "No commits found or invalid repo"}), 400

    df['cleaned'] = df['message'].apply(clean_text)
    X = vectorizer.transform(df['cleaned'])
    df['label'] = model.predict(X)

    # Save predictions
    filename = f"predicted_commits_from_github.xlsx"
    output_path = os.path.join(UPLOAD_FOLDER, filename)
    df[['message', 'label']].to_excel(output_path, index=False)

    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
