import pandas as pd
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=12)


def split(row):
    return " ".join(row)


def assign_label(tm):
    tm = round(tm / 10)
    if tm <= 0:
        return 0
    if 1 > tm > 0:
        return 1
    if 2 > tm > 1:
        return 2
    if 3 > tm > 2:
        return 3
    if 4 > tm > 3:
        return 4
    if 5 > tm > 4:
        return 5
    if 6 > tm > 5:
        return 6
    if 7 > tm > 6:
        return 7
    if 8 > tm > 7:
        return 8
    if 9 > tm > 8:
        return 9
    if 10 > tm > 9:
        return 10
    if 11 > tm > 10:
        return 11
    if 12 > tm > 11:
        return 12
    if tm > 12:
        return 13


def preprocess_raw_data():
    df = pd.read_csv('data/train.csv')
    df['label'] = kmeans.fit_predict(df[['tm']])
    df['protein_sequence'] = df['protein_sequence'].map(split)
    df = df.drop(['seq_id', 'pH', 'data_source', 'tm'], axis=1)
    df.to_csv('data/preprocessed_data.csv', index=False)


preprocess_raw_data()
