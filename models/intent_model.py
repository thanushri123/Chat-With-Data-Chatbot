# models/intent_model.py
"""
A tiny CNN that tells us whether the user wants a chart or a plain answer.
Everything is built from scratch – no pretrained embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------
# 1. Tokeniser (very forgiving)
# -------------------------------------------------
def tokenise(sentence: str):
    return re.findall(r"\w+", sentence.lower())

# -------------------------------------------------
# 2. Vocabulary builder
# -------------------------------------------------
class Vocabulary:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, texts, min_frequency=1):
        counter = Counter(token for txt in texts for token in tokenise(txt))
        self.index_to_word = [self.PAD_TOKEN, self.UNK_TOKEN] + [
            w for w, c in counter.items() if c >= min_frequency
        ]
        self.word_to_index = {w: i for i, w in enumerate(self.index_to_word)}

    def __len__(self):
        return len(self.index_to_word)

    def encode(self, sentence: str):
        return [self.word_to_index.get(tok, self.word_to_index[self.UNK_TOKEN])
                for tok in tokenise(sentence)]

# -------------------------------------------------
# 3. Torch Dataset
# -------------------------------------------------
class IntentDataset(Dataset):
    MAX_SEQ_LEN = 30

    def __init__(self, csv_path, vocab=None, label_to_id=None):
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.sentences = df["text"].tolist()
        self.labels = df["label"].tolist()

        self.vocab = vocab or Vocabulary(self.sentences)
        self.label_to_id = label_to_id or {
            lbl: idx for idx, lbl in enumerate(sorted(set(self.labels)))
        }
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

        self.samples = [
            (self.vocab.encode(s), self.label_to_id[l])
            for s, l in zip(self.sentences, self.labels)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.tensor(seq), torch.tensor(label)

    @staticmethod
    def collate(batch):
        sequences, labels = zip(*batch)
        sequences = [torch.tensor(s[: IntentDataset.MAX_SEQ_LEN]) for s in sequences]
        sequences = nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=0
        )
        labels = torch.stack(labels)
        return sequences, labels

# -------------------------------------------------
# 4. CNN model (3 different kernel sizes → classic TextCNN)
# -------------------------------------------------
class IntentCNN(nn.Module):
    EMBEDDING_DIM = 64
    FILTER_SIZES = [3, 4, 5]
    NUM_FILTERS = [64, 128, 128]
    DROPOUT = 0.5

    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, self.EMBEDDING_DIM, padding_idx=0)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(self.EMBEDDING_DIM, n_filters, kernel)
                for n_filters, kernel in zip(self.NUM_FILTERS, self.FILTER_SIZES)
            ]
        )
        self.dropout = nn.Dropout(self.DROPOUT)
        self.fc = nn.Linear(sum(self.NUM_FILTERS), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x).transpose(1, 2)           # (batch, emb, seq)
        conv_outputs = [
            F.relu(conv(embedded)).max(dim=2).values for conv in self.convs
        ]
        pooled = torch.cat(conv_outputs, dim=1)                # (batch, total_filters)
        dropped = self.dropout(pooled)
        logits = self.fc(dropped)
        return logits

# -------------------------------------------------
# 5. Persistence helpers
# -------------------------------------------------
def save_intent_model(model, vocab, label_to_id, path="models/intent.pt"):
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": vocab.index_to_word,
            "label_to_id": label_to_id,
        },
        path,
    )

def load_intent_model(path="models/intent.pt"):
    checkpoint = torch.load(path, map_location="cpu")
    vocab = Vocabulary([])
    vocab.index_to_word = checkpoint["vocab"]
    vocab.word_to_index = {w: i for i, w in enumerate(vocab.index_to_word)}

    model = IntentCNN(len(vocab), len(checkpoint["label_to_id"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocab, checkpoint["label_to_id"]
