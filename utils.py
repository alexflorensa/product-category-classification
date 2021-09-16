from pathlib import Path

import numpy as np
import pandas as pd
from typing import Union, Tuple

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from domain.dataset import CategoryClassificationDataset
from domain.tokenizer import Tokenizer


TRAIN_BATCH_SIZE = 256
VAL_BATCH_SIZE = 128


def load_data(file_path: Union[str, Path]) -> 'pd.DataFrame':
    df = pd.read_csv(file_path)
    df = df[df["language"] == "spanish"]
    df = df.drop(["label_quality", "language"], axis=1).reset_index(drop=True)
    return df


def build_dataloaders(
        data: 'pd.DataFrame',
        tokenizer: 'Tokenizer',
        train_size: float = 0.9
) -> Tuple['DataLoader', 'DataLoader']:
    mask = np.random.rand(len(data)) < train_size
    train_dataset = data[mask]
    val_dataset = data[~mask]

    train_dataset = CategoryClassificationDataset(train_dataset, tokenizer)
    val_dataset = CategoryClassificationDataset(val_dataset, tokenizer)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=TRAIN_BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=6)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=VAL_BATCH_SIZE,
                                num_workers=4)
    return train_dataloader, val_dataloader


def build_tokenizer(data: 'pd.DataFrame', n_most_common_words: int) -> 'Tokenizer':
    tokenizer = Tokenizer(num_words=n_most_common_words, oov_token="UK")
    tokenizer.fit_on_texts(np.asarray(data["title"]))
    return tokenizer


def build_category_encoder(data: 'pd.DataFrame') -> 'LabelEncoder':
    category_encoder = LabelEncoder()
    category_encoder.fit(data["category"].unique())
    data["category"] = category_encoder.transform(data["category"])
    return category_encoder
