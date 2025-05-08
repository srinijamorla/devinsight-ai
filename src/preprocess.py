import pandas as pd
import re

def label_commit(msg):
    msg = msg.lower()
    if 'fix' in msg or 'bug' in msg or 'error' in msg or 'issue' in msg:
        return 'bugfix'
    elif 'refactor' in msg or 'optimize' in msg:
        return 'refactor'
    elif 'add' in msg or 'implement' in msg or 'create' in msg or 'support' in msg:
        return 'feature'
    elif 'doc' in msg or 'readme' in msg or 'comment' in msg or 'typo' in msg:
        return 'docs'
    else:
        return 'other'

def apply_commit_labels(df):
    df['label'] = df['label'].fillna('').replace('', None)
    df['label'] = df.apply(
        lambda row: label_commit(row['message']) if pd.isna(row['label']) else row['label'],
        axis=1
    )
    return df

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text