import torch
from torch.utils.data import Dataset
import numpy as np
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    import pandas as pd
    from domain.tokenizer import Tokenizer


class CategoryClassificationDataset(Dataset):
    def __init__(
            self,
            df: 'pd.Dataframe',
            tokenizer: 'Tokenizer',
            title_length: int = 30  # median of sequences lengths
    ) -> None:
        super().__init__()
        self.df = df
        self.title_length = title_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple['torch.Tensor', ...]:
        title, category = self.df.iloc[idx]
        title = np.asarray(self.tokenizer.texts_to_sequences([title], length=self.title_length)).flatten()

        return torch.from_numpy(title), torch.tensor(category)
