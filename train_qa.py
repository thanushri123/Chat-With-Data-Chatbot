# train_qa.py
"""
Generate a few hundred synthetic questions and train the NER Bi-LSTM.
"""

import numpy as np
from models.qa_model import NERDataset, build_ner_model, save_ner_model

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Create a tiny synthetic corpus (feel free to expand)
    # ------------------------------------------------------------------
    base_questions = [
        "what is total sales",
        "show me average units by region",
        "give me a pie chart of sales by product",
        "sum of units in North",
        "average sales per product",
        "count of transactions in South",
    ]
    questions = base_questions * 50  # 300 examples

    # ------------------------------------------------------------------
    # 2. Dataset + model
    # ------------------------------------------------------------------
    dataset = NERDataset(questions, batch_size=32)
    model = build_ner_model(vocab_size=len(dataset.word_list), tag_count=len(dataset.tag_list))

    # ------------------------------------------------------------------
    # 3. Train (15 epochs is enough)
    # ------------------------------------------------------------------
    model.fit(dataset, epochs=15, verbose=1)

    # ------------------------------------------------------------------
    # 4. Persist
    # ------------------------------------------------------------------
    save_ner_model(model, dataset.word_list, dataset.tag_to_id)
    print("NER model saved → models/qa_model_*")
