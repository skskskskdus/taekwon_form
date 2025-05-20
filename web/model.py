import torch.nn as nn
from torch.optim import Adam
import torch

class PoseTransformer(nn.Module):
    def __init__(self, input_dim=66, num_classes=18):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 256)
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):  # x: (B, 33, 2)
        x = x.view(x.size(0), -1).unsqueeze(1)  # (B, 1, 66)
        x = self.embedding(x)                  # (B, 1, 256)
        x = self.encoder(x)                    # (B, 1, 256)
        x = x.mean(dim=1)                      # (B, 256)
        return self.fc(x)                      # (B, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)