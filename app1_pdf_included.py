import os
import glob
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import csv
import re
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.sparse import hstack

SAMPLE_FOLDER_TXT = "sample_resumes"
SAMPLE_FOLDER_PDF = "sample_resumes_pdf"
MODEL_FILE = "model.joblib"

SKILLS = ["python","sql","excel","tableau","power bi","pandas","numpy","scikit-learn",
          "matplotlib","seaborn","statistics","regression","etl","aws","bigquery","spark"]

def read_text_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except:
        return ""

def read_pdf_file(path):
    try:
        return extract_text(path) or ""
    except:
        return ""

def read_resume_text(path):
    path = str(path)
    if path.lower().endswith(".pdf"):
        return read_pdf_file(path)
    else:
        return read_text_file(path)

def read_uploaded_file_to_text(uploaded_file):
    # uploaded_file is a Streamlit UploadedFile-like object
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        # write to temp file and extract
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            txt = read_pdf_file(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
        return txt
    else:
        # assume text
        try:
            return uploaded_file.read().decode("utf-8")
        except:
            return ""

def extract_features_from_text(text):
    t = (text or "").lower()
    feat = {}
    for s in SKILLS:
        feat[f"skill_{s}"] = 1 if s in t else 0
    feat["count_skills"] = sum(feat[f"skill_{s}"] for s in SKILLS)
    years = 0
    m = re.search(r"(\d+)\s+years", t)
    if m:
        try:
            years = int(m.group(1))
        except:
            years = 0
    feat["years_exp"] = years
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
    files = sorted(glob.glob(os.path.join(folder, "*")))
    for p in files:
        if p.lower().endswith((".txt", ".pdf")):
            txt = read_resume_text(p)
            fname = os.path.basename(p)
            texts.append(txt)
            feats = extract_features_from_text(txt)
            feats["resume_id"] = fname
            rows.append(feats)
    if not rows:
        return pd.DataFrame(), [], []
    df = pd.DataFrame(rows).set_index("resume_id")
    return df, texts, [os.path.basename(p) for p in files if p.lower().endswith((".txt", ".pdf"))]

def train_and_save_model_from_folder(folder):
    gt_path = os.path.join(folder, "ground_truth.csv")
    if not os.path.exists(gt_path):
        st.warning(f"ground_truth.csv not found in {folder}. Generate samples first.")
        return None
    gt = pd.read_csv(gt_path)
    X_struct, texts, ids = build_dataset_from_folder(folder)
    if X_struct.empty:
        st.warning(f"No .txt or .pdf resumes found in {folder}.")
        return None
    # Align order of texts to gt resume_id
    # build a mapping from resume_id to text
    id_to_text = {}
    idx = 0
    for fid in X_struct.index:
        id_to_text[fid] = texts[idx]
        idx += 1
    texts_ordered = [id_to_text.get(rid, "") for rid in gt['resume_id']]
    X_struct = X_struct.reindex(gt['resume_id']).fillna(0)
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')
    X_text = tfidf.fit_transform(texts_ordered)
    X_struct_arr = X_struct.values.astype(float)
    X = hstack([X_text, X_struct_arr])
    y = gt['label'].values
    if len(set(y)) == 1:
        st.warning("Ground truth labels contain only one class; cannot train.")
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = LogisticRegression(solver='liblinear', max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f = f1_score(y_test, y_pred, zero_division=0)
    joblib.dump({"model": model, "tfidf": tfidf, "features": X_struct.columns.tolist()}, MODEL_FILE)
    return {"precision": p, "recall": r, "f1": f}

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
    # ensure all features present
    for f in features_order:
        if f not in struct_df.columns:
            struct_df[f] = 0
    X_struct_arr = struct_df[features_order].values.astype(float)
    X = hstack([X_text, X_struct_arr])
    prob = float(model.predict_proba(X)[:,1][0])
    present = [s for s in SKILLS if s in (text or "").lower()]
    reasons = f"skills: {', '.join(present[:6])}; years_exp={feats['years_exp']}; count_skills={feats['count_skills']}"
    return prob, reasons, feats

# --- Streamlit UI ---
st.set_page_config(page_title="Resume Screener (TXT+PDF)", layout="centered")
st.title("Resume Screener — TXT & PDF support")

st.markdown("Generate sample resumes (TXT or PDF), train model, then upload resumes (.txt or .pdf) to score them.")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Generate sample .txt resumes (50)"):
        import subprocess, sys
        subprocess.run([sys.executable, "gen_samples.py", "--n", "50"])
        st.success("Generated sample .txt resumes in ./sample_resumes")
with col2:
    if st.button("Generate sample .pdf resumes (50)"):
        import subprocess, sys
        subprocess.run([sys.executable, "gen_samples_pdf.py", "--n", "50"])
        st.success("Generated sample .pdf resumes in ./sample_resumes_pdf")
with col3:
    if st.button("Train model on all sample folders"):
        # prefer pdf folder if exists, else txt
        folder = SAMPLE_FOLDER_PDF if os.path.exists(SAMPLE_FOLDER_PDF) else SAMPLE_FOLDER_TXT
        res = train_and_save_model_from_folder(folder)
        if res:
            st.success(f"Trained model — precision={res['precision']:.2f}, recall={res['recall']:.2f}, f1={res['f1']:.2f}")
        else:
            st.error("Training failed. Ensure sample folder and ground_truth.csv exist.")

artefacts = load_model()
if artefacts is None:
    st.info("No trained model found. Train on generated samples (click 'Train model').")
else:
    st.success("Loaded trained model.")

uploaded = st.file_uploader("Upload .txt or .pdf resume files", accept_multiple_files=True, type=["txt","pdf"])
if uploaded:
    if artefacts is None:
        st.error("Train or generate model first.")
    else:
        scored = []
        for f in uploaded:
            text = read_uploaded_file_to_text(f)
            prob, reasons, feats = score_text_with_model(text, artefacts)
            scored.append({"filename": f.name, "score": prob, "reasons": reasons})
        df = pd.DataFrame(scored).sort_values("score", ascending=False)
        st.write("Top candidates:")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", data=csv, file_name="scores.csv", mime="text/csv")

st.markdown("Model file: model.joblib — contains model, tfidf, feature order.")
st.markdown("Notes: PDF extraction uses pdfminer.six (text PDFs only). For scanned PDFs add OCR (pytesseract). Use human review for final hiring decisions.")
