import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

from src.preprocess import clean_text
from src.model import load_model
from src.github_fetch import fetch_commits

app = Flask(__name__)
app.secret_key = 'devinsight-secret'  # In production, use a secure env var

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Dummy User setup
class User(UserMixin):
    def __init__(self, id):
        self.id = id
        self.name = "admin"
        self.password = "password"

users = {'admin': User(id=1)}

@login_manager.user_loader
def load_user(user_id):
    return users.get("admin") if user_id == "1" else None

# Uploads folder setup
current_dir = os.getcwd()
UPLOAD_FOLDER = os.path.join(current_dir, 'src', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model, vectorizer = load_model()

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'password':
            login_user(users['admin'])
            return redirect(url_for('index'))
        else:
            flash("Invalid credentials")
    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# Home
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# File prediction
@app.route('/predict', methods=['POST'])
@login_required
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

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        df['confidence'] = proba.max(axis=1)
    else:
        df['confidence'] = "N/A"

    output_path = os.path.join(UPLOAD_FOLDER, f"predicted_{filename}")
    df[['message', 'label', 'confidence']].to_excel(output_path, index=False)

    return send_file(output_path, as_attachment=True)

# GitHub repo analysis
@app.route('/analyze_repo', methods=['POST'])
@login_required
def analyze_repo():
    repo_url = request.form.get('repo_url')
    if not repo_url:
        return jsonify({"error": "No GitHub repo URL provided"}), 400

    df = fetch_commits(repo_url)
    if df.empty:
        return jsonify({"error": "No commits found or invalid repo"}), 400

    df['cleaned'] = df['message'].apply(clean_text)
    X = vectorizer.transform(df['cleaned'])
    df['label'] = model.predict(X)

    output_path = os.path.join(UPLOAD_FOLDER, f"predicted_commits_from_github.xlsx")
    df[['message', 'label']].to_excel(output_path, index=False)

    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
