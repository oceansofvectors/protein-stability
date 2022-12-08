from transformers import BertTokenizer
import pandas as pd
from tqdm import tqdm
from torch import LongTensor

tokenizer = BertTokenizer.from_pretrained("l-yohai/bigbird-roberta-base-mnli")


def tokenize(sequence: str) -> LongTensor:
    sequence = ' '.join(sequence)
    inputs = tokenizer(sequence)
    encoded_sequence = inputs["input_ids"]
    return LongTensor(encoded_sequence)


if __name__ == "__main__":

    df = pd.read_csv('data/train.csv')
    for index, row in tqdm(df.iterrows()):
        tokenize(row.get('protein_sequence'))
