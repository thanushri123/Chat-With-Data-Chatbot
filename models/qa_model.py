# models/qa_model.py
"""
Bi-LSTM NER that tags column names and aggregates in user questions.
Trained on synthetic BIO sentences – no pretrained weights.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import re
import json
import os
from tensorflow.keras import layers, models

# -------------------------------------------------
# 1. Tokeniser (same as PyTorch side)
# -------------------------------------------------
def tokenise(text: str):
    return re.findall(r"\w+", text.lower())

# -------------------------------------------------
# 2. Vocabulary for the NER model
# -------------------------------------------------
def build_vocab(sentences):
    from collections import Counter
    counter = Counter(t for s in sentences for t in tokenise(s))
    word_list = ["<pad>", "<unk>"] + [w for w, c in counter.items() if c >= 2]
    return word_list, {w: i for i, w in enumerate(word_list)}

# -------------------------------------------------
# 3. BIO tagging logic
# -------------------------------------------------
COLUMN_KEYWORDS = {"region", "product", "date", "units", "sales"}
AGGREGATE_KEYWORDS = {"total", "sum", "average", "avg", "count"}

def bio_tag_sentence(sentence: str):
    tokens = tokenise(sentence)
    tags = ["O"] * len(tokens)

    for i, tok in enumerate(tokens):
        if tok in AGGREGATE_KEYWORDS:
            tags[i] = "B-AGG" if i == 0 or tags[i - 1] != "I-AGG" else "I-AGG"
        elif tok in COLUMN_KEYWORDS:
            tags[i] = "B-COL" if i == 0 or tags[i - 1] != "I-COL" else "I-COL"
    return tokens, tags

# -------------------------------------------------
# 4. TF Dataset (Keras Sequence)
# -------------------------------------------------
class NERDataset(tf.keras.utils.Sequence):
    MAX_LEN = 20

    def __init__(self, questions, batch_size=32):
        self.questions = questions
        self.batch_size = batch_size
        self.word_list, self.word_to_id = build_vocab(questions)

        self.tag_list = ["O", "B-COL", "I-COL", "B-AGG", "I-AGG"]
        self.tag_to_id = {t: i for i, t in enumerate(self.tag_list)}
        self.id_to_tag = {i: t for t, i in self.tag_to_id.items()}

    def __len__(self):
        return int(np.ceil(len(self.questions) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.questions[idx * self.batch_size : (idx + 1) * self.batch_size]
        X, y = [], []
        for q in batch:
            toks, tags = bio_tag_sentence(q)
            ids = [self.word_to_id.get(t, 1) for t in toks[: self.MAX_LEN]]
            ids += [0] * (self.MAX_LEN - len(ids))
            tg = [self.tag_to_id[t] for t in tags[: self.MAX_LEN]]
            tg += [0] * (self.MAX_LEN - len(tg))
            X.append(ids)
            y.append(tg)
        return np.array(X), np.array(y)

# -------------------------------------------------
# 5. Model architecture
# -------------------------------------------------
def build_ner_model(vocab_size, tag_count, embedding_dim=64, lstm_units=128):
    inputs = layers.Input(shape=(None,), dtype="int32")
    emb = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs)
    bi_lstm = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(emb)
    outputs = layers.TimeDistributed(
        layers.Dense(tag_count, activation="softmax")
    )(bi_lstm)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

# -------------------------------------------------
# 6. Save / load helpers
# -------------------------------------------------
def save_ner_model(model, word_list, tag_to_id, folder="models/qa_model"):
    os.makedirs(folder, exist_ok=True)
    model.save_weights(f"{folder}_weights.h5")
    meta = {"word_list": word_list, "tag_to_id": tag_to_id}
    with open(f"{folder}_meta.json", "w") as f:
        json.dump(meta, f)

def load_ner_model(folder="models/qa_model"):
    with open(f"{folder}_meta.json") as f:
        meta = json.load(f)
    word_list, tag_to_id = meta["word_list"], meta["tag_to_id"]
    word_to_id = {w: i for i, w in enumerate(word_list)}
    model = build_ner_model(len(word_list), len(tag_to_id))
    model.load_weights(f"{folder}_weights.h5")
    return model, word_to_id, tag_to_id
