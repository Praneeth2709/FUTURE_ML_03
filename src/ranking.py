import os
import pandas as pd


def rank_candidates(df, scores):

    ranked = df.copy()

    # Raw cosine similarity (0-100)
    ranked["Similarity Score"] = scores * 100

    # Normalize so the top candidate scores 100
    max_score = ranked["Similarity Score"].max()

    if max_score > 0:
        ranked["ATS Score"] = (
            ranked["Similarity Score"] / max_score
        ) * 100
    else:
        ranked["ATS Score"] = 0

    ranked["ATS Score"] = ranked["ATS Score"].round(2)

    ranked = ranked.sort_values(
        by="ATS Score",
        ascending=False
    )

    ranked["Rank"] = range(1, len(ranked) + 1)

    os.makedirs("outputs", exist_ok=True)

    ranked.to_csv(
        "outputs/ranked_candidates.csv",
        index=False
    )

    return ranked