# FUTURE_ML_03
# 🤖 AI Resume Screening & Candidate Ranking Dashboard

An automated, ML-driven hiring decision-support platform designed to parse unstructured resume data, extract critical domain skills, evaluate profile alignments, and systematically rank candidate pipelines against dynamic job requirements. 

Built with **Python**, **Streamlit**, and **Natural Language Processing (NLP)**, this system automates high-volume candidate screening, reduces human workload, and mitigates subjectivity in the initial recruiting funnel.

---

## 🛠️ Key Architectural Features

* **NLP Normalization & Preprocessing**: Cleans and normalizes unstructured text blocks into tokenized formats ready for downstream vector processing.
* **Vector Space Modeling**: Leverages **TF-IDF Vectorization** alongside a custom-calibrated **Cosine Similarity Matrix** to accurately match profile context against active job descriptions.
* **Calibrated ATS Scoring System**: Computes and scales an alignment rating (0–100%) to provide relative fit indications clear of statistical noise.
* **Skill Gap Auditing**: Dynamically extracts required keywords to contrast matching attributes and list critical missing requirements side-by-side.
* **Interactive Intelligence Dashboard**: Includes full advanced profile filtering, real-time dataset search tools, interactive Plotly visualization splits, and download-ready pipeline exports.

---

## 📂 Project Directory Structure

Based on our active production repository workspace configuration:

```text
FUTURE_ML_03/
├── data/
│   ├── Resume.csv               # Baseline multi-category dataset source
│   └── cleaned_resume.csv       # Preprocessed runtime destination (ignored in versioning)
├── src/
│   ├── data_loader.py           # Streamlined dataset validation operations
│   ├── preprocess.py            # Custom regex & NLP tokenization pipelines
│   ├── skill_extractor.py       # Entity and keyword parsing utility engine
│   ├── similarity.py            # Cosine similarity and alignment math arrays
│   └── ranking.py               # Relative pipeline sequencing logic
├── app.py                       # Main Streamlit premium UI dashboard controller
├── requirements.txt             # Production dependency specification tree
└── .gitignore                   # Explicit staging file path exclusions

##Technical Stack & Core Tooling
Framework Interface: Streamlit (Premium Custom CSS Layout Configuration)

Mathematical Operations: NumPy & Pandas

Natural Language Processing: spaCy (en_core_web_sm) & NLTK

Machine Learning Backing: Scikit-Learn (Feature Extraction & Linear Similarities)

Analytical Visualization: Plotly Express (Distribution Histograms & Segment Volumetrics)
