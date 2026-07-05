import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from src.data_loader import load_dataset
from src.preprocess import clean_text
from src.skill_extractor import extract_skills
from src.similarity import rank_resumes
from src.ranking import rank_candidates

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening Dashboard",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI Styling
st.markdown("""
    <style>
    .main { background-color: #f9fafb; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; }
    div[data-testid="stMetricLabel"] { font-size: 0.95rem; font-weight: 600; color: #4B5563; }
    .stButton>button {
        background-color: #1E3A8A; color: white; border-radius: 6px; width: 100%;
        font-weight: 600; height: 3rem; border: none; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #2563EB; color: white; }
    .skill-tag {
        background-color: #EEF2F6; color: #1E3A8A; padding: 6px 12px; 
        border-radius: 20px; font-weight: 500; font-size: 13px; margin: 3px; display: inline-block;
        border: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Initialize Session State (Prevents UI Reset on Search)
# -------------------------------------------------
if "processed_pipeline" not in st.session_state:
    st.session_state.processed_pipeline = None
if "blueprint_skills" not in st.session_state:
    st.session_state.blueprint_skills = []

# -------------------------------------------------
# Professional Sidebar
# -------------------------------------------------
st.sidebar.title("🤖 AI Resume Screening")
st.sidebar.markdown("---")

st.sidebar.subheader("Engine Architecture")
st.sidebar.info("""
✅ **NLP Normalization**
✅ **TF-IDF Vector Space**
✅ **Cosine Similarity Matrix**
✅ **Calibrated ATS Scoring (0-100)**
""")

st.sidebar.markdown("---")
st.sidebar.subheader("Developer Workspace")
st.sidebar.success("Praneeth Varma")

# -------------------------------------------------
# Main Interface Header
# -------------------------------------------------
st.title("📄 AI Resume Screening & Candidate Ranking Dashboard")
st.caption("Rank candidate pipelines automatically using Natural Language Processing and realistic text-vector evaluations.")
st.markdown("---")

# -------------------------------------------------
# Input Component
# -------------------------------------------------
st.subheader("🎯 Target Profile Definition")
job_description = st.text_area(
    "Paste Target Job Specification / Requirements",
    height=230,
    placeholder="""We are looking for a Machine Learning Engineer with experience in:
Python
Machine Learning
SQL
TensorFlow
Git
Docker
AWS
Deep Learning
NLP"""
)

# -------------------------------------------------
# Processing Pipeline Execution Engine
# -------------------------------------------------
if st.button("🚀 Execute Talent Pipeline Match"):

    if job_description.strip() == "":
        st.warning("⚠️ Action Required: Please enter a valid Job Description context.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("🔄 Loading candidate resume database...")
        df = load_dataset("data/Resume.csv")
        progress_bar.progress(20)

        status_text.text("🔄 Executing text normalization and data cleaning...")
        df["Cleaned Resume"] = df["Resume_str"].apply(clean_text)
        cleaned_jd = clean_text(job_description)
        progress_bar.progress(45)

        status_text.text("🔄 Calculating vector match scores and fixing array dimension alignment...")
        raw_scores = rank_resumes(cleaned_jd, df["Cleaned Resume"].tolist())
        
        # Ensure 1D array profile similarity format
        raw_scores = np.array(raw_scores)
        if raw_scores.ndim > 1:
            raw_scores = raw_scores.ravel()[:len(df)] if raw_scores.shape[0] == len(df) else raw_scores[:, 0]
            
        # Realistic Calibration Optimization (Maps raw cosine values safely around 0.35 - 0.92)
        calibrated_scores = np.tanh(raw_scores * 1.8)
        
        # FIXED HERE: Passing as an explicit NumPy array so 'scores * 100' scales element-wise
        ranked = rank_candidates(df, calibrated_scores)
        progress_bar.progress(70)

        status_text.text("🔄 Auditing structural requirement skill gaps...")
        jd_skills = extract_skills(cleaned_jd)

        matched = []
        missing = []
        for resume in ranked["Cleaned Resume"]:
            resume_skills = extract_skills(resume)
            matched.append(", ".join(resume_skills))
            gaps = list(set(jd_skills) - set(resume_skills))
            missing.append(", ".join(sorted(gaps)))

        ranked["Matched Skills"] = matched
        ranked["Missing Skills"] = missing

        # Map score safely into calibrated ATS Rating formatting targets
        if "Similarity Score" in ranked.columns:
            # If your backend already multiplies by 100, read it directly
            ranked["ATS Score"] = ranked["Similarity Score"].round(1)
        else:
            score_col = [c for c in ranked.columns if 'score' in c.lower()][0]
            ranked["ATS Score"] = ranked[score_col].round(1)
            
        # Protect ceiling thresholds to remove computational text noise
        ranked["ATS Score"] = ranked["ATS Score"].clip(lower=0.0, upper=94.5)

        # Cache properties into active persistent session layout
        st.session_state.processed_pipeline = ranked
        st.session_state.blueprint_skills = jd_skills

        progress_bar.progress(100)
        status_text.empty()
        st.success("✅ Talent Pool Evaluation Completed Successfully!")
        st.balloons()

# -------------------------------------------------
# Evaluation Dashboard Output View Layout
# -------------------------------------------------
if st.session_state.processed_pipeline is not None:
    ranked = st.session_state.processed_pipeline
    jd_skills = st.session_state.blueprint_skills

    # KPI Summary Metric Cards
    st.markdown("### 📊 Pipeline Insights Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Resumes", f"{len(ranked)}")
    col2.metric("Highest ATS Match", f"{ranked['ATS Score'].max():.1f}%")
    col3.metric("Average Pool Fit", f"{ranked['ATS Score'].mean():.1f}%")
    col4.metric("Role Categories", f"{ranked['Category'].nunique()}")

    st.markdown("---")

    # Required Skills Extracted Section
    st.subheader("🛠 Required Skills Blueprint")
    if len(jd_skills):
        skills_html = "".join([f"<span class='skill-tag'>⚙️ {skill}</span>" for skill in jd_skills])
        st.markdown(skills_html, unsafe_allow_html=True)
    else:
        st.warning("No predefined profile skills extracted.")
    st.markdown("<br>", unsafe_allow_html=True)

    # Search Bar Filter
    st.subheader("🔍 Advanced Candidate Directory Filtering")
    search = st.text_input("Search Pool By Professional Category/Vertical", placeholder="e.g. Data Science, Developer, HR")

    filtered = ranked
    if search:
        filtered = ranked[ranked["Category"].str.contains(search, case=False, na=False)]

    # Dynamic Result Table View (Top 20 Matches)
    st.subheader("🏆 Top Ranked Candidates")
    display_df = filtered[[
        "Rank", "Category", "ATS Score", "Matched Skills", "Missing Skills"
    ]].head(20).copy()

    st.dataframe(
        display_df,
        column_config={
            "ATS Score": st.column_config.ProgressColumn(
                "ATS Score (0–100)",
                help="Calibrated structural vector representation fit index.",
                format="%.1f%%",
                min_value=0,
                max_value=100
            ),
            "Rank": st.column_config.NumberColumn("Rank Index", format="# %d")
        },
        use_container_width=True,
        hide_index=True
    )

    # Download Operations
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Full Selection Results",
        csv,
        "ranked_candidates.csv",
        "text/csv"
    )

    st.markdown("---")

    # Analytical Interactive Plotly Charts
    st.subheader("📈 Pipeline Performance & Category Concentration")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### **ATS Score Distribution Curve**")
        fig_hist = px.histogram(
            filtered,
            x="ATS Score",
            nbins=15,
            range_x=[0, 100],
            labels={"ATS Score": "Match Score (%)", "count": "Candidates Count"},
            color_discrete_sequence=['#1E3A8A'],
            template="plotly_white"
        )
        fig_hist.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with chart_col2:
        st.markdown("#### **Top Talent Pool Categories**")
        cat_data = filtered["Category"].value_counts().head(10).reset_index()
        cat_data.columns = ["Category", "Volume Count"]
        
        fig_bar = px.bar(
            cat_data,
            x="Volume Count",
            y="Category",
            orientation='h',
            color="Volume Count",
            color_continuous_scale=px.colors.sequential.Blugrn,
            labels={"Volume Count": "Count Total", "Category": "Professional Verticals"},
            template="plotly_white"
        )
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)