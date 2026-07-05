import pandas as pd


def load_dataset(file_path="data/Resume.csv"):

    df = pd.read_csv(file_path)

    df = df[["ID", "Resume_str", "Category"]]

    df.drop_duplicates(inplace=True)

    df.dropna(inplace=True)

    return df