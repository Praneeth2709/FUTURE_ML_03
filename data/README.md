# Dataset

This project uses the Resume Dataset from Kaggle.

Download:
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Place the downloaded `Resume.csv` file inside this folder.

After downloading, run:

```bash
python src/preprocess_dataset.py
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
🏗️ Technical Stack & Core ToolingFramework Interface: Streamlit (Premium Custom CSS Layout Configuration)Mathematical Operations: NumPy & PandasNatural Language Processing: spaCy (en_core_web_sm) & NLTKMachine Learning Backing: Scikit-Learn (Feature Extraction & Linear Similarities)Analytical Visualization: Plotly Express (Distribution Histograms & Segment Volumetrics)🚀 Local Deployment Setup InstructionsFollow these rapid configuration commands to boot the developer environment locally on your workstation:1. Clone the WorkspaceBashgit clone [https://github.com/YOUR_USERNAME/FUTURE_ML_03.git](https://github.com/YOUR_USERNAME/FUTURE_ML_03.git)
cd FUTURE_ML_03
2. Configure Your Virtual EnvironmentBash# Generate the isolated execution runtime environment
python -m venv venv

# Activate on Windows systems
venv\Scripts\activate

# Activate on macOS / Linux systems
source venv/bin/activate
3. Install Required DependenciesEnsure your package manager fetches explicit system footprints down from the requirements.txt file tree, including the downstream spaCy reference tags:Bashpip install -r requirements.txt
python -m spacy download en_core_web_sm
4. Execute the Application EngineBashstreamlit run app.py
📊 Evaluation & Metrics LogicThis decision-support framework analyzes text arrays based on normalized structural overlap. The baseline profile metric maps vector similarity scores safely via a hyperbolic tangent calibration layer ($ \tanh $ scaling optimization) to securely establish valid, readable structural ranges:$$ATS\ Score = \tanh(\text{Raw Cosine Score} \times 1.8) \times 100$$This safeguards candidate ranking boundaries, isolates background computational text noises, and provides actionable HR metrics tracking for technical reviewers and startup recruitment stakeholders alike.👤 Developer SpaceLead Engineer: Praneeth VarmaProject Track: Future Interns Machine Learning Cohort Assignment Task