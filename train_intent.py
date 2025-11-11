# train_intent.py
"""
Train the tiny IntentCNN on the hand-crafted intents.csv.
Runs in < 2 minutes on a laptop CPU.
"""

import torch
import torch.optim as optim
import torch.nn as nn
from models.intent_model import IntentDataset, IntentCNN, save_intent_model

if __name__ == "__main__":
    # 1. Load data & build vocab + label map
    dataset = IntentDataset("data/intents.csv")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=IntentDataset.collate
    )

    # 2. Model, optimiser, loss
    model = IntentCNN(vocab_size=len(dataset.vocab), num_classes=len(dataset.label_to_id))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 3. Training loop (12 epochs is plenty)
    for epoch in range(1, 13):
        model.train()
        epoch_loss = 0.0
        for sequences, labels in loader:
            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch:02d} – loss {epoch_loss/len(loader):.4f}")

    # 4. Persist
    save_intent_model(model, dataset.vocab, dataset.label_to_id)
    print("Intent model saved → models/intent.pt")
