import torch
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification
from torch import LongTensor
from tokenizer import tokenize
import pandas as pd
from tqdm import tqdm

labels = [n for n in range(0, 10)]


class Model:
    def __init__(self):
        self.model = BigBirdForSequenceClassification.from_pretrained("l-yohai/bigbird-roberta-base-mnli")
        self.train_df = pd.read_csv('data/train.csv')

    def determine_label(self, tm: int) -> LongTensor:
        if tm >= 13:
            tm = 13
        return LongTensor([1 if n == round(tm/10) else 0 for n in range(0, 13)])

    def train(self):
        for index, row in tqdm(self.train_df.iterrows()):
            sequence = row.get('protein_sequence')
            print(sequence)
            input_ids = tokenize(sequence)
            _labels = self.determine_label(row.get('tm'))
            self.forward(input_ids, _labels)

    def forward(self, input_ids: LongTensor, _labels: LongTensor):
        self.model.forward(input_ids=input_ids, labels=_labels)


if __name__ == "__main__":
    model = Model()
    model.train()
