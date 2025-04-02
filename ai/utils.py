# Updated ai/utils.py
import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import json
import time
import matplotlib.pyplot as plt
from game.state import TileType, BOARD_SIZE


# Helper functions that might be imported from training.py
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


def get_ai_move(model, state, temperature=0.0, deterministic=True, mcts_simulations=0, opening_book=None):
    """
    Get the best move for the AI based on the current state, with optional MCTS and opening book.

    Args:
        model: The neural network model
        state: Current game state
        temperature: Temperature parameter for controlling exploration
        deterministic: If True, always choose the highest probability move
        mcts_simulations: If > 0, use MCTS with this many simulations
        opening_book: Optional opening book to use

    Returns:
        Tuple of (chosen_move, value) where value is the model's evaluation of the position
    """
    # Try opening book first if provided
    if opening_book is not None:
        book_move = opening_book.get_move(state, temperature=temperature)
        if book_move:
            # We found a move in the opening book
            return book_move, 0.0  # No value estimate for book moves

    # Use MCTS if requested
    if mcts_simulations > 0:
        from ai.mcts import mcts_search
        move, _ = mcts_search(
            model,
            state,
            num_simulations=mcts_simulations,
            temperature=temperature,
            dirichlet_noise=not deterministic
        )

        # Get value estimate with a simple forward pass
        state_tensor = torch.tensor(state.encode_for_network(), dtype=torch.float32).unsqueeze(0)
        device = next(model.parameters()).device
        state_tensor = state_tensor.to(device)

        with torch.no_grad():
            _, value = model(state_tensor)

        return move, value.item()

    # Otherwise use direct model prediction
    # Get valid moves
    valid_moves = state.get_valid_moves()

    if not valid_moves:
        return None, 0.0  # No valid moves

    # Convert valid moves to indices
    valid_indices = [action_to_index(move) for move in valid_moves]

    # Get model prediction
    state_tensor = torch.tensor(state.encode_for_network(), dtype=torch.float32).unsqueeze(0)
    device = next(model.parameters()).device
    state_tensor = state_tensor.to(device)

    with torch.no_grad():
        policy_logits, value = model(state_tensor)

    # Convert to probability distribution over valid moves
    policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

    # Filter for valid moves
    valid_policy = np.zeros_like(policy)
    for idx in valid_indices:
        valid_policy[idx] = policy[idx]

    # Normalize
    if valid_policy.sum() > 0:
        valid_policy /= valid_policy.sum()
    else:
        # If all valid moves have zero probability, use uniform distribution
        for idx in valid_indices:
            valid_policy[idx] = 1.0 / len(valid_indices)

    # Apply temperature
    if not deterministic and temperature != 1.0 and temperature > 0:
        # Adjust probabilities based on temperature
        valid_policy = valid_policy ** (1.0 / temperature)
        valid_policy = valid_policy / np.sum(valid_policy)  # Renormalize

    # Choose move
    if deterministic:
        chosen_idx = np.argmax(valid_policy)
    else:
        # Sample from the probability distribution
        chosen_idx = np.random.choice(len(policy), p=valid_policy)

    # Convert index back to move
    chosen_move = None
    for i, idx in enumerate(valid_indices):
        if idx == chosen_idx:
            chosen_move = valid_moves[i]
            break

    if chosen_move is None:
        # Fallback - should rarely happen
        chosen_move = valid_moves[0]
        print("Warning: Failed to map chosen index to valid move. Using fallback.")

    return chosen_move, value.item()


def evaluate_model(model, num_games=50, opponent=None, opening_book=None, mcts_simulations=0, verbose=True):
    """
    Evaluate a model by playing against another model or random moves.

    Args:
        model: Model to evaluate
        num_games: Number of games to play
        opponent: Another model or "random" or "mcts" or None (random)
        opening_book: Optional opening book to use
        mcts_simulations: Number of MCTS simulations to use (if > 0)
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation metrics
    """
    from game.state import SkudPaiShoState

    if isinstance(opponent, str) and opponent.lower() == "random":
        opponent = None  # Use random moves

    wins = 0
    losses = 0
    draws = 0
    game_lengths = []
    harmony_counts = []

    if verbose:
        print(f"Evaluating over {num_games} games...")

    start_time = time.time()

    for game_num in range(num_games):
        state = SkudPaiShoState()

        # Track the moves to avoid infinite loops
        move_history = []

        while not state.is_game_over() and state.turn_number < 200:
            current_player = state.current_player

            if current_player == 1:  # Model 1's turn
                move, _ = get_ai_move(
                    model,
                    state,
                    temperature=0.0,
                    deterministic=True,
                    mcts_simulations=mcts_simulations,
                    opening_book=opening_book
                )
            else:  # Opponent's turn
                if opponent is not None and isinstance(opponent, torch.nn.Module):
                    # Opponent is another model
                    move, _ = get_ai_move(
                        opponent,
                        state,
                        temperature=0.0,
                        deterministic=True,
                        mcts_simulations=mcts_simulations,
                        opening_book=opening_book
                    )
                elif opponent == "mcts":
                    # Use MCTS without a model (pure search)
                    # This is slower but can be a good baseline
                    from ai.mcts import mcts_search
                    move, _ = mcts_search(model, state, num_simulations=mcts_simulations, temperature=0.0)
                else:
                    # Random opponent
                    valid_moves = state.get_valid_moves()
                    if not valid_moves:
                        break
                    move = random.choice(valid_moves)

            if move is None:
                # No valid moves
                break

            # Apply the move
            state.make_move(move)

            # Add to history to detect repetitions
            move_history.append(move)

            # Check for move repetition (3-fold repetition = draw)
            if len(move_history) >= 12:
                last_4_moves = move_history[-4:]
                previous_4_moves = move_history[-8:-4]
                earlier_4_moves = move_history[-12:-8]

                if last_4_moves == previous_4_moves == earlier_4_moves:
                    # Three-fold repetition detected
                    if verbose:
                        print("Draw by repetition detected")
                    draws += 1
                    break

        # Record game results
        game_lengths.append(state.turn_number)
        harmony_counts.append(state.count_harmonies())

        if state.is_game_over():
            result = state.get_reward(1)  # From player 1's perspective
            if result > 0:
                wins += 1
            elif result < 0:
                losses += 1
            else:
                draws += 1
        else:
            # Draw by move limit or repetition
            if draws == game_num:  # If not already counted as draw by repetition
                draws += 1

        if verbose and (game_num + 1) % 10 == 0:
            win_rate = wins / (game_num + 1)
            elapsed = time.time() - start_time
            time_per_game = elapsed / (game_num + 1)
            remaining = time_per_game * (num_games - game_num - 1)

            print(f"Game {game_num + 1}/{num_games}: Win rate = {win_rate:.3f}, "
                  f"ETA: {remaining:.1f}s")

    results = {
        "win_rate": wins / num_games,
        "loss_rate": losses / num_games,
        "draw_rate": draws / num_games,
        "avg_game_length": sum(game_lengths) / len(game_lengths) if game_lengths else 0,
        "avg_harmony_count": sum(harmony_counts) / len(harmony_counts) if harmony_counts else 0,
        "total_games": num_games
    }

    if verbose:
        print(f"Final results:")
        print(f"  Win rate: {results['win_rate']:.3f}")
        print(f"  Loss rate: {results['loss_rate']:.3f}")
        print(f"  Draw rate: {results['draw_rate']:.3f}")
        print(f"  Average game length: {results['avg_game_length']:.1f} moves")
        print(f"  Average harmony count: {results['avg_harmony_count']:.2f}")

    return results


def analyze_position(model, state, top_k=5, use_mcts=True, mcts_simulations=800):
    """
    Analyze a position and return the top moves with probabilities.

    Args:
        model: The neural network model
        state: Current game state
        top_k: Number of top moves to return
        use_mcts: Whether to use MCTS for deeper analysis
        mcts_simulations: Number of MCTS simulations if use_mcts=True

    Returns:
        Tuple of (position_value, list of (move, probability) tuples)
    """
    # Get valid moves
    valid_moves = state.get_valid_moves()

    if not valid_moves:
        return 0.0, []  # No valid moves

    if use_mcts:
        from ai.mcts import MCTSNode, mcts_search

        # Run MCTS
        move, policy = mcts_search(
            model,
            state,
            num_simulations=mcts_simulations,
            temperature=0.0,
            dirichlet_noise=False,
            return_details=True  # Get detailed results
        )

        # Get move probabilities from the returned policy
        move_probs = []
        for m in valid_moves:
            idx = action_to_index(m)
            if idx < len(policy):
                move_probs.append((m, policy[idx]))
            else:
                move_probs.append((m, 0.0))

        # Sort moves by probability
        move_probs.sort(key=lambda x: x[1], reverse=True)

        # Get value estimate
        state_tensor = torch.tensor(state.encode_for_network(), dtype=torch.float32).unsqueeze(0)
        device = next(model.parameters()).device
        state_tensor = state_tensor.to(device)

        with torch.no_grad():
            _, value = model(state_tensor)

        return value.item(), move_probs[:top_k]
    else:
        # Use direct model prediction
        state_tensor = torch.tensor(state.encode_for_network(), dtype=torch.float32).unsqueeze(0)
        device = next(model.parameters()).device
        state_tensor = state_tensor.to(device)

        with torch.no_grad():
            policy_logits, value = model(state_tensor)

        # Convert to probability distribution
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        # Convert valid moves to indices
        valid_indices = [action_to_index(move) for move in valid_moves]

        # Get probabilities for valid moves
        move_probs = [(move, policy[idx]) for move, idx in zip(valid_moves, valid_indices)]

        # Sort by probability (descending)
        move_probs.sort(key=lambda x: x[1], reverse=True)

        return value.item(), move_probs[:top_k]


def create_initial_model(input_channels, architecture="A1", device=None):
    """
    Create and initialize a fresh model with specific architecture.

    Args:
        input_channels: Number of input channels
        architecture: Which architecture to use (A1, A2, A3, A4, A5 or "best")
        device: Device to place model on (cpu, cuda)

    Returns:
        Initialized model
    """
    from ai.model import SkudPaiShoTransformer

    # Architecture parameters based on your table
    architectures = {
        "A1": {"d_model": 256, "nhead": 8, "num_layers": 3, "dropout": 0.2},
        "A2": {"d_model": 256, "nhead": 4, "num_layers": 9, "dropout": 0.1},
        "A3": {"d_model": 256, "nhead": 8, "num_layers": 6, "dropout": 0.1},
        "A4": {"d_model": 128, "nhead": 4, "num_layers": 3, "dropout": 0.1},
        "A5": {"d_model": 512, "nhead": 8, "num_layers": 6, "dropout": 0.1},
        # "best" is A1 based on your experiments
        "best": {"d_model": 256, "nhead": 8, "num_layers": 3, "dropout": 0.2}
    }

    # Use specified architecture or default to best
    params = architectures.get(architecture, architectures["best"])

    # Create model
    model = SkudPaiShoTransformer(
        input_channels=input_channels,
        d_model=params["d_model"],
        nhead=params["nhead"],
        num_layers=params["num_layers"],
        dropout=params["dropout"]
    )

    # Move to device if specified
    if device is not None:
        model = model.to(device)

    print(f"Created model with architecture {architecture}:")
    print(f"  Input channels: {input_channels}")
    print(f"  Model dimension: {params['d_model']}")
    print(f"  Attention heads: {params['nhead']}")
    print(f"  Layers: {params['num_layers']}")
    print(f"  Dropout: {params['dropout']}")

    return model


def visualize_game(game_states, model=None, save_path=None, show=True):
    """
    Visualize a game as a sequence of board states with optional model evaluation.

    Args:
        game_states: List of game states
        model: Optional model to provide position evaluation
        save_path: Path to save visualization
        show: Whether to display the visualization
    """
    num_states = len(game_states)
    rows = (num_states + 3) // 4  # Ceiling division to get number of rows

    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    if rows == 1:
        axes = [axes]  # Make it 2D for consistent indexing

    for i, state in enumerate(game_states):
        row, col = i // 4, i % 4
        ax = axes[row][col]

        # Simple board visualization (customize based on your game representation)
        board = state.board  # Adjust based on your state representation

        # Simple text representation for now - replace with proper visualization
        ax.text(0.5, 0.5, f"Turn {state.turn_number}\nPlayer {state.current_player}",
                ha='center', va='center', fontsize=12)

        # Add model evaluation if provided
        if model is not None:
            state_tensor = torch.tensor(state.encode_for_network(), dtype=torch.float32).unsqueeze(0)
            device = next(model.parameters()).device
            state_tensor = state_tensor.to(device)

            with torch.no_grad():
                _, value = model(state_tensor)

            eval_text = f"Eval: {value.item():.2f}"
            ax.text(0.5, 0.2, eval_text, ha='center', va='center', fontsize=10)

        ax.set_title(f"Move {i + 1}")
        ax.axis('off')

    # Hide unused subplots
    for i in range(num_states, rows * 4):
        row, col = i // 4, i % 4
        axes[row][col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def play_game_vs_ai(model, use_mcts=True, mcts_simulations=400, opening_book=None, player_side=1):
    """
    Play a game against the AI interactively.

    Args:
        model: The neural network model
        use_mcts: Whether to use MCTS for AI moves
        mcts_simulations: Number of MCTS simulations if use_mcts=True
        opening_book: Optional opening book to use
        player_side: Which side the human player plays (1 or 2)

    Returns:
        Final game state
    """
    from game.state import SkudPaiShoState

    state = SkudPaiShoState()
    game_history = []

    print("Starting a new game!")
    print("Enter moves in the format: 'plant fire 3 4' or 'move 3 4 5 6'")

    while not state.is_game_over() and state.turn_number < 200:
        game_history.append(state.copy())

        print(f"\nTurn {state.turn_number}, Player {state.current_player}'s turn")
        print(state)  # Assuming your state class has a string representation

        if state.current_player == player_side:
            # Human player's turn
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                print("No valid moves available. Game over.")
                break

            print("Valid moves:")
            for i, move in enumerate(valid_moves):
                print(f"{i + 1}: {move}")

            # Get human input
            while True:
                try:
                    choice = input("Enter move number or move description: ")

                    # Check if input is a move number
                    if choice.isdigit() and 1 <= int(choice) <= len(valid_moves):
                        move = valid_moves[int(choice) - 1]
                        break

                    # Otherwise parse as move description
                    parts = choice.split()
                    if parts[0] == "plant":
                        # Parse plant move: "plant fire 3 4"
                        tile_type = getattr(TileType, parts[1].upper())
                        x, y = int(parts[2]), int(parts[3])
                        move = ("plant", tile_type, x, y)
                    elif parts[0] == "move":
                        # Parse move move: "move 3 4 5 6"
                        from_x, from_y = int(parts[1]), int(parts[2])
                        to_x, to_y = int(parts[3]), int(parts[4])
                        move = ("move", from_x, from_y, to_x, to_y)
                    else:
                        raise ValueError("Invalid move format")

                    # Check if move is valid
                    if move in valid_moves:
                        break
                    else:
                        print("Invalid move. Try again.")

                except (ValueError, IndexError, AttributeError) as e:
                    print(f"Error parsing move: {e}")
                    print("Try again or enter a move number.")
        else:
            # AI's turn
            print("AI is thinking...")
            move, value = get_ai_move(
                model,
                state,
                temperature=0.0,
                deterministic=True,
                mcts_simulations=mcts_simulations if use_mcts else 0,
                opening_book=opening_book
            )
            print(f"AI plays: {move} (evaluation: {value:.2f})")

        # Make the move
        state.make_move(move)

    # Game over
    game_history.append(state.copy())
    print("\nGame over!")
    print(state)

    result = state.get_reward(player_side)
    if result > 0:
        print("You win!")
    elif result < 0:
        print("AI wins!")
    else:
        print("It's a draw!")

    return game_history