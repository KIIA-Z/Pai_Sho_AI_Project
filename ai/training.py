# ai_old/training.py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from game.state import SkudPaiShoState, TileType, BOARD_SIZE


def action_to_index(action, board_size=BOARD_SIZE):
    """Convert a game action to an index in the policy array."""
    num_positions = board_size * board_size
    num_tile_types = len(TileType) - 1  # Excluding EMPTY

    if action[0] == "plant":
        _, tile_type, x, y = action
        # Index for planting tiles
        tile_idx = tile_type.value - 1  # -1 because we skip EMPTY (0)
        pos_idx = y * board_size + x
        return tile_idx * num_positions + pos_idx

    elif action[0] == "move":
        _, from_x, from_y, to_x, to_y = action
        # Index for moving tiles (after all planting indices)
        from_idx = from_y * board_size + from_x
        to_idx = to_y * board_size + to_x
        plant_indices = num_tile_types * num_positions
        return plant_indices + from_idx * num_positions + to_idx


def index_to_action(index, board_size=BOARD_SIZE):
    """Convert an index in the policy array to a game action."""
    num_positions = board_size * board_size
    num_tile_types = len(TileType) - 1  # Excluding EMPTY
    plant_indices = num_tile_types * num_positions

    if index < plant_indices:
        # This is a plant action
        tile_idx = index // num_positions
        pos_idx = index % num_positions
        x = pos_idx % board_size
        y = pos_idx // board_size
        tile_type = TileType(tile_idx + 1)  # +1 because we skip EMPTY (0)
        return ("plant", tile_type, x, y)
    else:
        # This is a move action
        move_index = index - plant_indices
        from_idx = move_index // num_positions
        to_idx = move_index % num_positions
        from_x = from_idx % board_size
        from_y = from_idx // board_size
        to_x = to_idx % board_size
        to_y = to_idx // board_size
        return ("move", from_x, from_y, to_x, to_y)


class SkudPaiShoDataset(Dataset):
    def __init__(self, game_records):
        """Initialize dataset from game records."""
        self.states = []
        self.policies = []
        self.values = []

        for state, policy, value in game_records:
            self.states.append(state.encode_for_network())
            self.policies.append(policy)
            self.values.append(value)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.states[idx], dtype=torch.float32),
            torch.tensor(self.policies[idx], dtype=torch.float32),
            torch.tensor(self.values[idx], dtype=torch.float32)
        )


def self_play(model, num_games=100, exploration=True):
    """Generate self-play games for training."""
    game_records = []

    for game_num in range(num_games):
        print(f"Playing self-play game {game_num + 1}/{num_games}")
        state = SkudPaiShoState()
        game_history = []

        while not state.is_game_over() and state.turn_number < 200:  # Max 200 moves
            # Get valid moves
            valid_moves = state.get_valid_moves()

            if not valid_moves:
                break  # No valid moves, end the game

            # Convert valid moves to indices
            valid_indices = [action_to_index(move) for move in valid_moves]

            # Get model prediction
            state_tensor = torch.tensor(state.encode_for_network(), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                policy_logits, _ = model(state_tensor)

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

            # Add noise for exploration
            if exploration:
                valid_policy = 0.75 * valid_policy + 0.25 * np.random.dirichlet([0.3] * len(valid_policy))


                valid_policy = valid_policy / np.sum(valid_policy)  # Ensure it sums to exactly 1

                # And add a safety check:
                if not np.isclose(np.sum(valid_policy), 1.0):
                    # Handle edge case - maybe just use uniform distribution
                    valid_policy = np.ones(len(valid_policy)) / len(valid_policy)

            # Choose move
            if exploration:
                chosen_idx = np.random.choice(len(policy), p=valid_policy)
            else:
                chosen_idx = np.argmax(valid_policy)

            # Convert index back to move
            chosen_move = None
            for i, idx in enumerate(valid_indices):
                if idx == chosen_idx:
                    chosen_move = valid_moves[i]
                    break

            if chosen_move is None:
                chosen_move = valid_moves[0]  # Fallback

            # Store state and policy
            game_history.append((state.copy(), valid_policy))

            # Make move
            state.make_move(chosen_move)

        # Game over, determine outcome
        player1_reward = state.get_reward(1)

        # Add outcome to all states in game
        for past_state, policy in game_history:
            # Adjust outcome based on player (1 for win, -1 for loss, 0 for draw)
            player_reward = player1_reward if past_state.current_player == 1 else -player1_reward
            game_records.append((past_state, policy, player_reward))

    return game_records


def train(model, game_records, epochs=5, batch_size=128, lr=0.001, return_losses=False):
    """
    Train the network on game records.

    Args:
        model: The neural network model
        game_records: List of (state, value, policy) tuples from self-play
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        return_losses: Whether to return the final losses (added parameter)

    Returns:
        Tuple of (policy_loss, value_loss) if return_losses is True
    """
    # Create a training dataset from game records
    states = []
    values = []
    policies = []

    for state, policy, value in game_records:
        states.append(state.encode_for_network())
        policies.append(policy)
        values.append(value)

    # Convert to tensors
    states = torch.tensor(np.array(states), dtype=torch.float32)
    values = torch.tensor(np.array(values), dtype=torch.float32)
    policies = torch.tensor(np.array(policies), dtype=torch.float32)

    # Create data loader
    dataset = torch.utils.data.TensorDataset(states, policies, values)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss functions
    value_criterion = torch.nn.MSELoss()

    # Use KL divergence for policy loss instead of CrossEntropyLoss
    # This handles probability distributions better
    def policy_criterion(output, target):
        return F.kl_div(
            F.log_softmax(output, dim=1),
            target,
            reduction='batchmean'
        )

    # Train
    model.train()
    device = next(model.parameters()).device

    policy_losses = []
    value_losses = []

    for epoch in range(epochs):
        running_policy_loss = 0.0
        running_value_loss = 0.0

        for i, (state_batch, policy_batch, value_batch) in enumerate(data_loader):
            # Move to device
            state_batch = state_batch.to(device)
            policy_batch = policy_batch.to(device)
            value_batch = value_batch.to(device)

            # Forward pass
            policy_output, value_output = model(state_batch)

            # Debug prints (optional - can be removed in production)
            if i == 0 and epoch == 0:
                print("Policy output shape:", policy_output.shape)
                print("Policy batch shape:", policy_batch.shape)
                print("Policy output dtype:", policy_output.dtype)
                print("Policy batch dtype:", policy_batch.dtype)
                print("Policy output min/max:", torch.min(policy_output).item(), torch.max(policy_output).item())

            # Calculate losses
            policy_loss = policy_criterion(policy_output, policy_batch)
            value_loss = value_criterion(value_output.squeeze(), value_batch)

            # Combined loss
            loss = policy_loss + value_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running losses
            running_policy_loss += policy_loss.item()
            running_value_loss += value_loss.item()

        # Calculate average losses for this epoch
        avg_policy_loss = running_policy_loss / len(data_loader)
        avg_value_loss = running_value_loss / len(data_loader)

        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")

    # Return final losses
    final_policy_loss = policy_losses[-1] if policy_losses else 0.0
    final_value_loss = value_losses[-1] if value_losses else 0.0

    if return_losses:
        return final_policy_loss, final_value_loss
    else:
        return final_policy_loss, final_value_loss  # For backward compatibility, always return losses


def train_skud_pai_sho_ai(model, iterations=50, games_per_iteration=20, epochs_per_iteration=5):
    """Main training loop using iterative self-play."""
    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/{iterations}")

        # Generate self-play games
        game_records = self_play(model, num_games=games_per_iteration)

        # Train model on new data
        train(model, game_records, epochs=epochs_per_iteration)

        # Save checkpoint
        torch.save(model.state_dict(), f"models/skud_pai_sho_model_iter_{iteration}.pth")

    return model