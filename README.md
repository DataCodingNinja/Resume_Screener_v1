# Resume Screener (Streamlit UI )

Lightweight resume parsing + fit scorer demo using TF-IDF + Logistic Regression.
Designed to run on a modest PC (8GB). All files can stay in one folder.

Files
- app.py — Streamlit front-end, trains model on sample resumes and scores uploads.
- gen_samples.py — Generates synthetic .txt resumes and ground_truth.csv.
- sample_resumes/ — (created by generator) contains resumes and ground_truth.csv.
- model.joblib — produced after training.
- requirements.txt — Python dependencies.

Quick setup
1. Create folder and save app.py, gen_samples.py, requirements.txt.
2. Create virtual environment and install:
   python -m venv venv
   source venv/bin/activate   # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt

Generate sample data
- Run: python gen_samples.py --n 50
  This creates sample_resumes/ and sample_resumes/ground_truth.csv

Train model
- From VS Code or terminal run the Streamlit app:
  streamlit run app.py
- Or in the app click "Train model on sample_resumes".

Use the app
- Upload one or more plain .txt resume files to get a score and short reasons.
- Download scored results as CSV.

Notes & limitations
- Demo uses synthetic data and simple heuristics; not production-ready.
- Only handles plain text resumes (.txt). PDF support requires adding pdf text-extraction.
- Risk of bias — use as an assistive tool with human review.
- For production, replace synthetic labels with real annotated data and add validation.

- MIT (use as you like).

