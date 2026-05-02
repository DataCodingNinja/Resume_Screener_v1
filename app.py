import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

SAMPLE_FOLDER = "sample_resumes"
MODEL_FILE = "model.joblib"

# simple skill list
SKILLS = ["python","sql","excel","tableau","power bi","pandas","numpy","scikit-learn",
          "matplotlib","seaborn","statistics","regression","etl","aws","bigquery","spark"]

def read_resume_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""

def extract_features_from_text(text):
    t = text.lower()
    feat = {}
    # skill presence
    for s in SKILLS:
        feat[f"skill_{s}"] = 1 if s in t else 0
    # count skills
    feat["count_skills"] = sum(feat[f"skill_{s}"] for s in SKILLS)
    # years experience heuristic
    years = 0
    import re
    m = re.search(r"(\d+)\s+years", t)
    if m:
        years = int(m.group(1))
    feat["years_exp"] = years
    # education level heuristic
    edu = 0
    if "m.sc" in t or "master" in t or "msc" in t:
        edu = 2
    elif "bachelor" in t or "b.sc" in t or "bsc" in t:
        edu = 1
    feat["edu_level"] = edu
    return feat

def build_dataset_from_folder(folder):
    texts = []
    rows = []
    for p in glob.glob(os.path.join(folder, "*.txt")):
        txt = read_resume_text(p)
        fname = os.path.basename(p)
        texts.append(txt)
        feats = extract_features_from_text(txt)
        feats["resume_id"] = fname
        rows.append(feats)
    df = pd.DataFrame(rows).set_index("resume_id")
    return df, texts, [os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.txt"))]

def train_and_save_model(folder=SAMPLE_FOLDER):
    # read ground truth
    gt_path = os.path.join(folder, "ground_truth.csv")
    if not os.path.exists(gt_path):
        st.warning("ground_truth.csv not found in sample folder. Please run gen_samples.py first.")
        return None
    gt = pd.read_csv(gt_path)
    X_struct, texts, ids = build_dataset_from_folder(folder)
    # ensure alignment
    X_struct = X_struct.reindex(gt['resume_id']).fillna(0)
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')
    X_text = tfidf.fit_transform(texts)
    # combine: simple approach — convert struct to array and hstack with text
    from scipy.sparse import hstack
    X_struct_arr = X_struct.values.astype(float)
    X = hstack([X_text, X_struct_arr])
    y = gt['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = LogisticRegression(solver='liblinear', max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f = f1_score(y_test, y_pred, zero_division=0)
    joblib.dump({"model":model, "tfidf":tfidf, "features": X_struct.columns.tolist()}, MODEL_FILE)
    return {"precision":p, "recall":r, "f1":f}

def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)

def score_text_with_model(text, artefacts):
    feats = extract_features_from_text(text)
    struct_df = pd.DataFrame([feats])
    tfidf = artefacts["tfidf"]
    model = artefacts["model"]
    features_order = artefacts["features"]
    X_text = tfidf.transform([text])
    X_struct_arr = struct_df[features_order].values.astype(float)
    from scipy.sparse import hstack
    X = hstack([X_text, X_struct_arr])
    prob = model.predict_proba(X)[:,1][0]
    # create explanation: top present skills
    present = [s for s in SKILLS if s in text.lower()]
    reasons = f"skills: {', '.join(present[:5])}; years_exp={feats['years_exp']}; count_skills={feats['count_skills']}"
    return prob, reasons

# Streamlit UI
st.set_page_config(page_title="Resume Screener", layout="centered")
st.title("Resume Screener — Demo")

st.markdown("Upload one or more plain .txt resumes, or use generated sample resumes.")

col1, col2 = st.columns(2)
with col1:
    if st.button("Generate sample resumes (50)"):
        import subprocess, sys
        subprocess.run([sys.executable, "gen_samples.py", "--n", "50"])
        st.success("Generated sample resumes in ./sample_resumes")

with col2:
    if st.button("Train model on sample_resumes"):
        res = train_and_save_model()
        if res:
            st.success(f"Trained model — precision={res['precision']:.2f}, recall={res['recall']:.2f}, f1={res['f1']:.2f}")
        else:
            st.error("Training failed. Ensure sample_resumes/ground_truth.csv exists.")

artefacts = load_model()
if artefacts is None:
    st.info("No trained model found. Click 'Train model' to train on generated samples.")
else:
    st.success("Loaded trained model.")

uploaded = st.file_uploader("Upload .txt resume files", accept_multiple_files=True, type=["txt"])
if uploaded:
    if artefacts is None:
        st.error("Train or generate model first.")
    else:
        scored = []
        for f in uploaded:
            txt = f.read().decode("utf-8")
            prob, reasons = score_text_with_model(txt, artefacts)
            scored.append({"filename": f.name, "score": float(prob), "reasons": reasons})
        df = pd.DataFrame(scored).sort_values("score", ascending=False)
        st.write("Top candidates:")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", data=csv, file_name="scores.csv", mime="text/csv")

st.markdown("Model file: model.joblib — contains model, tfidf, feature order.")
st.markdown("Notes: This is a demo. Use human review for final decisions.")
