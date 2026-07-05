import pandas as pd

from src.preprocess import clean_text

print("Loading dataset...")

df = pd.read_csv("data/Resume.csv")

df = df[["ID", "Resume_str", "Category"]]

print("Cleaning resumes...")

df["Cleaned Resume"] = df["Resume_str"].apply(clean_text)

df.to_csv(
    "data/cleaned_resume.csv",
    index=False
)

print("Finished!")