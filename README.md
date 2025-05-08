# 💡 DevInsight AI

A Smart Insight Engine for Developer Productivity — powered by NLP and Machine Learning.  
Automatically **classify**, **tag**, and **summarize** Git commits to improve code understanding, documentation, and team workflows.

---

## 🚀 Features

| Feature | Description |
|--------|-------------|
| 🧠 **Commit Classification** | Classify commits as `feature`, `bugfix`, `refactor`, etc. using a trained ML model |
| 📁 **CSV Upload Support** | Upload your own commit data for instant predictions |
| 🌐 **Flask Web UI** | User-friendly interface for non-tech stakeholders |
| 📊 **Export Results** | Download labeled predictions in Excel format |
| 📈 **Notebook-Based EDA** | Visualize trends and insights using Jupyter |

---

## 🔍 Use Case

Developers and engineering teams often struggle to track what type of work is being done across large codebases. DevInsight AI solves this by:

- Automatically tagging and classifying work types
- Flagging areas with frequent bug fixes or refactors
- Summarizing intent from commit history
- Helping teams triage and document better

---

## 🧠 Tech Stack

| Area | Tools |
|------|-------|
| Core ML | `scikit-learn`, `TF-IDF`, `pandas`, `joblib` |
| NLP | `spaCy`, `NLTK` |
| Web App | `Flask`, `HTML`, `Bootstrap` |
| Visualization | `matplotlib`, `seaborn`, `Jupyter` |
| Deployment | Local Flask server or Streamlit (future) |

---

## 📁 Project Structure


devinsight-ai/
│
├── data/                         # CSV datasets (labeled/unlabeled commits)
├── notebooks/                   # Jupyter Notebooks for EDA
├── src/                         # ML model, preprocessing logic
│   ├── model.py
│   ├── preprocess.py
│   └── model.pkl                # Saved model
├── uploads/                     # Prediction input/output Excel files
├── webapp/                      # Flask app
│   ├── app.py
│   ├── templates/index.html
│   └── static/
├── README.md
├── requirements.txt
├── .gitignore
└── main.py                      # CLI/entry script (optional)
🛠 How to Run Locally
Clone the repo

git clone https://github.com/<your-username>/devinsight-ai.git
cd devinsight-ai
Create virtual environment

python3 -m venv .venv
source .venv/bin/activate
Install dependencies


pip install -r requirements.txt
Start the web app


python webapp/app.py
Visit: http://localhost:5000 in your browser

✨ Sample Predictions
Input: Unlabeled CSV of commit messages

Output: Predictions like:

Commit Message	Predicted Label
fix login issue in auth module	bugfix
add support for new user roles	feature
refactor validation logic	refactor

🔮 Planned Features
🔌 GitHub API Integration: Fetch commit history live from public/private repos

🧠 Vector Similarity Search: Find similar commits from history

✍️ LLM Summarization: Auto-generate human-readable summaries

📊 Dashboard: Trends of work types over time (optional extension)

📄 License
This project is licensed under the MIT License.

👤 Author
Srinija Morla
Backend Engineer | ML Enthusiast | Builder of DevInsight AI 🚀
LinkedIn | GitHub

🙌 Acknowledgements
spaCy
scikit-learn
Flask
GitHub Commit Datasets