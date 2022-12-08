import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import TrainingArguments, Trainer, BigBirdForSequenceClassification, \
    BigBirdTokenizer
from sklearn.model_selection import train_test_split

tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

print("Tokenizing dataset")
df = pd.read_csv('data/preprocessed_data.csv', nrows=100)
df['label'] = df['label'].astype('category').cat.codes
train_texts = list(df['protein_sequence'])
train_labels = list(df['label'])
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
train_encodings = tokenizer(train_texts, max_length=500, padding='max_length', truncation=True)
val_encodings = tokenizer(val_texts, max_length=500, padding='max_length', truncation=True)


class ProteinDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = ProteinDataset(train_encodings, train_labels)
val_dataset = ProteinDataset(val_encodings, val_labels)

model = BigBirdForSequenceClassification.from_pretrained("google/bigbird-roberta-base")

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)
print("Training")
trainer.train()
