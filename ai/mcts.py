# ai/mcts.py
import math
import numpy as np
import torch
import torch.nn.functional as F
import time
from collections import defaultdict
import random

from game.state import TileType, BOARD_SIZE
from ai.training import action_to_index, index_to_action

# Constants for MCTS
C_PUCT = 1.0  # Exploration constant
DIRICHLET_ALPHA = 0.3  # Alpha parameter for Dirichlet noise
DIRICHLET_WEIGHT = 0.25  # Weight of Dirichlet noise


class MCTSNode:
    """Node in the Monte Carlo Tree Search."""

    def __init__(self, state, parent=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.prior = prior  # P(s,a) from policy network
        self.children = {}  # Map from action to MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False

    def expand(self, policy):
        """Expand the node using the provided policy."""
        valid_moves = self.state.get_valid_moves()
        valid_indices = [action_to_index(move) for move in valid_moves]

        # Apply Dirichlet noise to root node for exploration
        if self.parent is None and len(valid_moves) > 0:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(valid_indices))
            policy_with_noise = {idx: (1 - DIRICHLET_WEIGHT) * policy[idx] + DIRICHLET_WEIGHT * noise[i]
                                 for i, idx in enumerate(valid_indices)}
        else:
            policy_with_noise = {idx: policy[idx] for idx in valid_indices}

        # Create children
        for i, move in enumerate(valid_moves):
            idx = valid_indices[i]
            child_state = self.state.copy()
            child_state.make_move(move)
            self.children[move] = MCTSNode(
                state=child_state,
                parent=self,
                prior=policy_with_noise[idx]
            )

        self.expanded = True

    def select_child(self):
        """Select a child node according to the PUCT algorithm."""
        # Check if there are any children
        if not self.children:
            return None, None

        # Find child with highest UCB score
        best_score = -float('inf')
        best_child = None
        best_action = None

        # Total visit count for parent
        sum_n = sum(child.visit_count for child in self.children.values())

        # Small constant to avoid division by zero
        c = 1e-8

        for action, child in self.children.items():
            # UCB formula with PUCT
            ucb_score = child.get_value() + C_PUCT * child.prior * math.sqrt(sum_n) / (1 + child.visit_count)

            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                best_action = action

        # Fallback if no best child was found (shouldn't happen if children exist)
        if best_child is None and self.children:
            # Just take the first child
            best_action = next(iter(self.children.keys()))
            best_child = self.children[best_action]

        return best_action, best_child


    def backup(self, value):
        """Update value and visit count for this node and all parents."""
        self.value_sum += value
        self.visit_count += 1

        if self.parent:
            # Flip value for parent (opponent's perspective)
            self.parent.backup(-value)

    def get_value(self):
        """Get the average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self):
        """Check if this node has been expanded."""
        return self.expanded

    def get_best_move(self, temperature=0.0):
        """
        Get the best move from this node based on visit counts.

        Args:
            temperature: Controls exploration vs exploitation
                temperature=0 means greedy selection (highest visit count)
                temperature>0 means probabilistic selection
        """
        if not self.children:
            return None

        # Get visit counts for all children
        visits = np.array([child.visit_count for child in self.children.values()])
        actions = list(self.children.keys())

        if temperature == 0:
            # Greedy selection
            best_idx = np.argmax(visits)
            return actions[best_idx]
        else:
            # Temperature-based selection
            visits = visits ** (1.0 / temperature)
            visits = visits / np.sum(visits)  # Normalize to probabilities
            selected_idx = np.random.choice(len(actions), p=visits)
            return actions[selected_idx]

    def get_move_probabilities(self, temperature=1.0):
        """
        Get probability distribution over moves based on visit counts.
        Used for training data generation.
        """
        visits = np.array([child.visit_count for child in self.children.values()])
        actions = list(self.children.keys())

        if temperature == 0:
            # One-hot distribution at the best move
            probs = np.zeros(len(actions))
            probs[np.argmax(visits)] = 1.0
            return actions, probs
        else:
            # Temperature-based distribution
            visits = visits ** (1.0 / temperature)
            probs = visits / np.sum(visits)
            return actions, probs


def mcts_search(model, state, num_simulations=800, temperature=1.0, dirichlet_noise=True, return_details=False):
    """
    Run Monte Carlo Tree Search using the neural network for guidance.

    Args:
        model: Neural network model providing policy and value predictions
        state: Current game state
        num_simulations: Number of MCTS simulations to run
        temperature: Temperature for final move selection
        dirichlet_noise: Whether to add Dirichlet noise at the root for exploration
        return_details: Whether to return additional details

    Returns:
        Best move according to search, policy probabilities for training
    """
    # Create root node
    root = MCTSNode(state)

    # Track time for debugging
    start_time = time.time()
    last_progress_report = start_time
    progress_interval = 5.0  # Report progress every 5 seconds

    # Run simulations
    for sim in range(num_simulations):
        # Periodic progress reporting for long-running searches
        current_time = time.time()
        if current_time - last_progress_report > progress_interval:
            elapsed = current_time - start_time
            print(
                f"MCTS progress: {sim}/{num_simulations} simulations ({sim / num_simulations * 100:.1f}%) in {elapsed:.1f} seconds")
            last_progress_report = current_time

        node = root
        search_path = [node]

        # Selection: traverse the tree to a leaf node
        while node is not None and node.is_expanded() and not node.state.is_game_over():
            action, node = node.select_child()
            # Break the loop if select_child returned None
            if node is None:
                break
            search_path.append(node)

        # Skip this simulation if node became None
        if node is None:
            continue

        # Check if leaf node is terminal
        if node.state.is_game_over():
            # Use the game result as value
            value = node.state.get_reward(node.state.current_player)
        else:
            # Expansion and evaluation: use neural network
            state_tensor = torch.tensor(node.state.encode_for_network(), dtype=torch.float32).unsqueeze(0)

            # Add debugging info for first simulation only
            if sim == 0:
                print(f"MCTS state tensor shape: {state_tensor.shape}")

            # Check for channel mismatch and correct if needed
            expected_channels = model.input_channels
            actual_channels = state_tensor.shape[1]

            if actual_channels != expected_channels:
                if sim == 0:  # Only print once
                    print(f"WARNING: Channel mismatch - model expects {expected_channels}, but got {actual_channels}")
                # Option 1: If there's one extra channel we can drop it (usually safer)
                if actual_channels == expected_channels + 1:
                    state_tensor = state_tensor[:, :expected_channels, :, :]
                    if sim == 0:  # Only print once
                        print(f"Dropping extra channel. New shape: {state_tensor.shape}")
                # Option 2: If we're missing a channel, we need a different fix

            device = next(model.parameters()).device
            state_tensor = state_tensor.to(device)

            try:
                with torch.no_grad():
                    policy_logits, value = model(state_tensor)

                # Convert policy logits to probabilities
                policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

                # Expand node with policy
                node.expand(policy)

                value = value.item()

            except Exception as e:
                # Provide detailed error information
                print(f"ERROR in MCTS neural network evaluation: {str(e)}")
                print(f"  State tensor shape: {state_tensor.shape}")
                print(f"  Model input channels: {model.input_channels}")
                print(f"  Device: {device}")
                # Choose a default value
                value = 0.0
                # Create a uniform policy for this node
                valid_moves = node.state.get_valid_moves()
                policy = np.ones(model.policy_head[-1].out_features) / model.policy_head[-1].out_features
                node.expand(policy)
                # Exit this simulation
                search_path[-1].backup(value)
                continue

        # Backup: update statistics for all nodes in search path
        search_path[-1].backup(value)

    # Report final timing
    total_time = time.time() - start_time
    print(
        f"MCTS completed {num_simulations} simulations in {total_time:.2f} seconds ({num_simulations / total_time:.1f} sims/sec)")

    # Return best move based on visit counts and the policy for training
    best_move = root.get_best_move(temperature)

    # Get policy for training (visit count distribution)
    actions, probs = root.get_move_probabilities(temperature=1.0)

    # Convert to neural network policy format
    policy = np.zeros(model.policy_head[-1].out_features)
    for action, prob in zip(actions, probs):
        idx = action_to_index(action)
        policy[idx] = prob

    return best_move, policy


def self_play_with_mcts(model, num_games=100, mcts_simulations=800, temperature_schedule=None):
    """
    Generate self-play games using MCTS.

    Args:
        model: Neural network model
        num_games: Number of games to play
        mcts_simulations: Number of MCTS simulations per move
        temperature_schedule: Dict mapping move number thresholds to temperatures
                             e.g., {0: 1.0, 30: 0.5, 60: 0.25}

    Returns:
        List of (state, policy, value) tuples for training
    """
    if temperature_schedule is None:
        # Default temperature schedule: start with exploration, then become more greedy
        temperature_schedule = {0: 1.0, 30: 0.5, 60: 0.0}

    game_records = []

    for game_num in range(num_games):
        print(f"Playing MCTS self-play game {game_num + 1}/{num_games}")
        state = state_class()
        game_history = []

        move_num = 0
        while not state.is_game_over() and move_num < 200:  # Max 200 moves to prevent infinite games
            # Determine temperature based on move number
            temperature = 1.0
            for threshold, temp in sorted(temperature_schedule.items()):
                if move_num >= threshold:
                    temperature = temp

            # Use MCTS to select move and get improved policy
            best_move, improved_policy = mcts_search(
                model,
                state,
                num_simulations=mcts_simulations,
                temperature=temperature
            )

            if best_move is None:
                break  # No valid moves

            # Store state and improved policy
            game_history.append((state.copy(), improved_policy))

            # Make move
            state.make_move(best_move)
            move_num += 1

        # Game over, determine outcome
        if state.is_game_over():
            player1_reward = state.get_reward(1)

            # Add outcome to all states in game
            for past_state, policy in game_history:
                # Value target is from perspective of player who just moved
                player_reward = player1_reward if past_state.current_player == 1 else -player1_reward
                game_records.append((past_state, policy, player_reward))

    return game_records


# Set the state class - replace with your actual state class
state_class = None  # This should be set to your game state class


def set_state_class(cls):
    """Set the state class for MCTS."""
    global state_class
    state_class = cls