import torch


def init_weights(m) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if isinstance(m, torch.nn.Embedding):
        torch.nn.init.xavier_uniform_(m.weight)


class CategoryClassification(torch.nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            n_classes: int,
            embedding_dim: int = 256,
            dropout: float = 0.1,
            lstm_hidden_size: int = 128
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.lstm = torch.nn.GRU(input_size=embedding_dim,
                                 hidden_size=lstm_hidden_size,
                                 bidirectional=True)
        self.head = torch.nn.Linear(in_features=256, out_features=n_classes)
        self.apply(init_weights)

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = torch.mean(x, axis=1)
        x = self.head(x)
        return x
