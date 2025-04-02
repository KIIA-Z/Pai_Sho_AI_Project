# ai_deprecated/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game.state import TileType, BOARD_SIZE


class SkudPaiShoTransformer(nn.Module):
    def __init__(self, input_channels, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        super(SkudPaiShoTransformer, self).__init__()

        # Initial convolutional layer to process the board state
        self.conv1 = nn.Conv2d(input_channels, d_model, kernel_size=3, padding=1)

        # Position encoding for the board
        self.pos_encoding = self.create_position_encoding(d_model, BOARD_SIZE)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Number of possible actions (plant or move for each position)
        # We'll use a simplified action space:
        # - For planting: tile_type * board_positions
        # - For moving: from_pos * to_pos
        # This is a simplification, would need refinement for actual implementation
        action_size = (len(TileType) - 1) * (BOARD_SIZE * BOARD_SIZE) + (BOARD_SIZE * BOARD_SIZE) * (
                    BOARD_SIZE * BOARD_SIZE)

        # Policy head (outputs move probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )

        # Value head (outputs state evaluation)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()  # Output between -1 and 1
        )

    def create_position_encoding(self, d_model, board_size):
        """Create positional encodings for the board."""
        pe = torch.zeros(board_size, board_size, d_model)
        for i in range(board_size):
            for j in range(board_size):
                for k in range(0, d_model, 2):
                    div_term = torch.exp(torch.tensor(k * -(np.log(10000.0) / d_model)))
                    pe[i, j, k] = torch.sin(((i * board_size + j) * div_term).clone().detach())
                    if k + 1 < d_model:
                        pe[i, j, k + 1] = torch.cos(((i * board_size + j) * div_term).clone().detach())
        return pe

    def forward(self, x):
        """Forward pass through the network."""
        batch_size = x.size(0)

        # Initial convolution
        x = F.relu(self.conv1(x))  # [batch_size, d_model, board_size, board_size]

        # Add positional encoding
        x = x.permute(0, 2, 3, 1)  # [batch_size, board_size, board_size, d_model]
        x = x + self.pos_encoding.to(x.device)
        x = x.permute(0, 3, 1, 2)  # [batch_size, d_model, board_size, board_size]

        # Reshape for transformer
        x = x.view(batch_size, x.size(1), -1)  # [batch_size, d_model, board_size*board_size]
        x = x.permute(0, 2, 1)  # [batch_size, board_size*board_size, d_model]

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, d_model]

        # Policy and value heads
        policy = self.policy_head(x)
        value = self.value_head(x)

        return policy, value