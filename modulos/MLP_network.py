import torch.nn as nn

class MLPBinaria(nn.Module):
    def __init__(self, input_dim, hidden_dim=10):
        super().__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)

        self.gelu = nn.GELU()

        self.dropout = nn.Dropout(0.3)

        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        out = self.fc(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.output(out)
        return out