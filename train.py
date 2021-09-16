from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from models import CategoryClassification
from utils import load_data, build_category_encoder, build_tokenizer, build_dataloaders
import gc 
import psutil
BASE_PATH = Path(".")

LEARNING_RATE = 0.001
TRAIN_SIZE = 0.9
EPOCHS = 30
N_MOST_COMMON_WORDS = 150_000


def main() -> int:
    data_path = BASE_PATH / "train.csv"
    if not data_path.exists():
        print(f"{data_path} does not exist")
        return 1

    data = load_data(data_path)
    category_encoder = build_category_encoder(data)
    tokenizer = build_tokenizer(data, N_MOST_COMMON_WORDS)
    train_dataloader, val_dataloader = build_dataloaders(data, tokenizer)

    n_classes = len(category_encoder.classes_)
    model = CategoryClassification(num_embeddings=N_MOST_COMMON_WORDS, n_classes=n_classes)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0
    acc = 0
    for epoch in range(1, EPOCHS + 1):
        model = model.train()
        train_bar = tqdm(train_dataloader)
        train_loss = []
        for sequences, labels in train_bar:
            sequences = sequences.cuda()
            labels = labels.cuda()
            outputs = model(sequences)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().item())
            train_bar.set_description(f"Loss={np.mean(train_loss):.5f}")

        model = model.eval()
        n = 0
        correct = 0
        for sequences, labels in val_dataloader:
            sequences = sequences.cuda()
            labels = labels.cuda()
            with torch.no_grad():
                outputs = model(sequences)

            outputs = torch.argmax(outputs, dim=1)
            correct += torch.sum(outputs == labels)
            n += len(labels)

        acc = correct / n
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"weights_{epoch}_{acc:.5f}.pth")
        print(f"\nEpoch {epoch}/{EPOCHS}: Acc={acc:.5f}")

    torch.save(model.state_dict(), f"weights_final_{acc:.5f}.pth")
    return 0


if __name__ == '__main__':
    exit(main())
