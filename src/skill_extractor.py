skills = [
    "python",
    "java",
    "c",
    "c++",
    "sql",
    "mysql",
    "postgresql",
    "mongodb",
    "excel",
    "power bi",
    "tableau",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "tensorflow",
    "keras",
    "pytorch",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "nlp",
    "computer vision",
    "opencv",
    "flask",
    "django",
    "streamlit",
    "react",
    "node.js",
    "node",
    "html",
    "css",
    "javascript",
    "git",
    "github",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "linux",
    "data analysis",
    "data science"
]


def extract_skills(text):

    text = text.lower()

    found = []

    for skill in skills:

        if skill in text:

            found.append(skill)

    return sorted(set(found))