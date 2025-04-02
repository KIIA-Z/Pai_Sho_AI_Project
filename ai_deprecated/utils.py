# ai_deprecated/utils.py
import torch
import torch.nn.functional as F
import numpy as np
from game.state import TileType, BOARD_SIZE
from ai_deprecated.training import action_to_index


def get_ai_move(model, state):
    """Get the best move for the AI based on the current state."""
    # Get valid moves
    valid_moves = state.get_valid_moves()

    if not valid_moves:
        return None  # No valid moves

    # Convert valid moves to indices
    valid_indices = [action_to_index(move) for move in valid_moves]

    # Get model prediction
    state_tensor = torch.tensor(state.encode_for_network(), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        policy_logits, value = model(state_tensor)

    # Convert to probability distribution over valid moves
    policy = F.softmax(policy_logits, dim=1).squeeze(0).numpy()

    # Filter for valid moves
    valid_policy = np.zeros_like(policy)
    for i, idx in enumerate(valid_indices):
        valid_policy[idx] = policy[idx]

    # Normalize
    if valid_policy.sum() > 0:
        valid_policy /= valid_policy.sum()
    else:
        # If all valid moves have zero probability, use uniform distribution
        for idx in valid_indices:
            valid_policy[idx] = 1.0 / len(valid_indices)

    # Choose best move
    chosen_idx = np.argmax(valid_policy)

    # Convert index back to move
    chosen_move = None
    for i, idx in enumerate(valid_indices):
        if idx == chosen_idx:
            chosen_move = valid_moves[i]
            break

    if chosen_move is None:
        chosen_move = valid_moves[0]  # Fallback

    return chosen_move, value.item()


def create_initial_model():
    """Create and initialize a fresh model."""
    from ai_deprecated.model import SkudPaiShoTransformer

    # Calculate input channels based on encoding
    # All tile types (except EMPTY) + turn + player + harmonies(2) + board mask
    input_channels = len(TileType) - 1 + 4

    # Debug print to verify
    print(f"Creating model with {input_channels} input channels")

    model = SkudPaiShoTransformer(input_channels)
    return model