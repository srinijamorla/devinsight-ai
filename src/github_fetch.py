# src/github_fetch.py

import os
import requests
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import urlparse
import streamlit as st

# Load OAuth token from .env file
load_dotenv()
GITHUB_TOKEN= ["REMOVED"]

headers = {
    "Authorization": f"token {GITHUB_TOKEN}"
}

def extract_repo_info(repo_url):
    """
    Extracts owner and repo name from a GitHub URL
    """
    path = urlparse(repo_url).path.strip("/")
    parts = path.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    else:
        raise ValueError("Invalid GitHub repository URL format.")

def fetch_commits(repo_url, per_page=100, max_pages=5):
    owner, repo = extract_repo_info(repo_url)
    commits = []

    for page in range(1, max_pages + 1):
        url = f"https://api.github.com/repos/{owner}/{repo}/commits"
        params = {
            "per_page": per_page,
            "page": page
        }
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Error fetching commits: {response.status_code}")
            break

        data = response.json()

        for commit in data:
            message = commit['commit']['message']
            author = commit['commit']['author']['name']
            date = commit['commit']['author']['date']
            commits.append({
                "author": author,
                "date": date,
                "message": message
            })

    return pd.DataFrame(commits)

def save_commits_to_csv(df, path="data/fetched_commits.csv"):
    df.to_csv(path, index=False)
    print(f"[✔] Saved {len(df)} commits to {path}")


if __name__ == "__main__":
    # You can hardcode the repo URL for testing or use input()
    default_url = "https://github.com/srinijamorla/chatbot_repo"
    user_input = input(f"Enter GitHub repo URL (default: {default_url}): ").strip()
    repo_url = user_input if user_input else default_url

    df = fetch_commits(repo_url)
    save_commits_to_csv(df)