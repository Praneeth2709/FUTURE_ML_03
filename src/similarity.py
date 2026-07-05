from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank_resumes(job_description, resumes):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )

    documents = [job_description] + resumes

    tfidf = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf[0:1],
        tfidf[1:]
    )

    return similarity.flatten()