# Updated ai/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game.state import TileType, BOARD_SIZE


class SkudPaiShoTransformer(nn.Module):
    def __init__(self, input_channels=20, d_model=256, nhead=8, num_layers=3, dropout=0.2, layer_norm_eps=1e-5):
        super(SkudPaiShoTransformer, self).__init__()

        self.input_channels = input_channels
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Initial convolutional layers with batch normalization for better gradient flow
        self.conv1 = nn.Conv2d(input_channels, d_model // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(d_model // 2)
        self.conv2 = nn.Conv2d(d_model // 2, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(d_model)

        # Position encoding for the board
        self.pos_encoding = self.create_position_encoding(d_model, BOARD_SIZE)

        # Improved transformer encoder with pre-layer normalization for stability
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,  # Wider feedforward network
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-normalization for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Calculate action space size (needs to match mcts.py)
        action_size = (len(TileType) - 1) * (BOARD_SIZE * BOARD_SIZE) + (BOARD_SIZE * BOARD_SIZE) * (
                    BOARD_SIZE * BOARD_SIZE)

        # Policy head with layer normalization and dropout
        self.policy_head = nn.Sequential(
            nn.LayerNorm(d_model, eps=layer_norm_eps),
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, action_size)
        )

        # Value head with layer normalization and dropout
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model, eps=layer_norm_eps),
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),  # Less dropout in deeper layers
            nn.Linear(256, 1),
            nn.Tanh()  # Output between -1 and 1
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization for better training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

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
        return nn.Parameter(pe, requires_grad=False)  # Fixed positional encoding

    def forward(self, x, return_attention=False):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            return_attention: If True, returns attention weights for visualization

        Returns:
            Tuple of (policy_logits, value) and optionally attention weights
        """
        batch_size = x.size(0)
        attention_weights = []

        # Initial convolution with batch norm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Add positional encoding
        x = x.permute(0, 2, 3, 1)  # [batch_size, board_size, board_size, d_model]
        x = x + self.pos_encoding.to(x.device)

        # Reshape for transformer: [batch_size, sequence_length, d_model]
        x = x.reshape(batch_size, BOARD_SIZE * BOARD_SIZE, self.d_model)

        # Store attention weights if requested
        if return_attention:
            # We need a custom forward pass through the transformer to capture attention
            for layer in self.transformer_encoder.layers:
                # Pre-norm is applied in the layer itself (norm_first=True)
                attn_output, attn_weights = layer.self_attn(
                    x, x, x,
                    need_weights=True,
                    average_attn_weights=False  # Get weights for each head
                )
                attention_weights.append(attn_weights)

                # Apply the rest of the layer operations
                x = x + layer.dropout1(attn_output)
                x = x + layer.dropout2(layer.linear2(layer.dropout(F.relu(layer.linear1(layer.norm2(x))))))
        else:
            # Standard transformer pass
            x = self.transformer_encoder(x)

        # Modified global attention-weighted pooling to fix dimension issues
        # Create a context vector with proper dimensions
        context_vector = x.mean(dim=1, keepdim=True)  # [batch_size, 1, d_model]

        # Compute attention scores - ensure dimensions match
        if context_vector.size(-1) == x.size(-1):
            # Use proper attention mechanism with compatible dimensions
            attention_scores = torch.bmm(x, context_vector.transpose(1, 2))  # [batch_size, seq, 1]
            attention_probs = F.softmax(attention_scores, dim=1)
            x = torch.bmm(attention_probs.transpose(1, 2), x).squeeze(1)  # [batch_size, d_model]
        else:
            # Fallback to simple mean pooling if dimensions don't match
            print(f"Warning: Dimension mismatch in attention. x: {x.size()}, context: {context_vector.size()}")
            x = x.mean(dim=1)  # [batch_size, d_model]

        # Policy and value heads
        policy = self.policy_head(x)
        value = self.value_head(x)

        if return_attention:
            return policy, value, attention_weights
        else:
            return policy, value

    def get_output_size(self):
        """Return the size of the policy output (action space size)."""
        # This is useful for MCTS to know the exact size of the policy vector
        return self.policy_head[-1].out_features