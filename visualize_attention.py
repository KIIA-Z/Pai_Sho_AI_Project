#!/usr/bin/env python
# visualize_attention.py - Script for visualizing attention patterns of the Skud Pai Sho AI

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import time

from game.state import SkudPaiShoState, TileType, BOARD_SIZE
from ai.model import SkudPaiShoTransformer
from ai.utils import get_ai_move


def modify_model_for_attention(model):
    """Modify the model to return attention weights."""
    # Only modify if not already modified
    if not hasattr(model, '_original_forward'):
        # Save original forward method
        model._original_forward = model.forward

        # Define a new forward method that returns attention weights
        def forward_with_attention(x, return_attention=False):
            if return_attention:
                try:
                    return model._original_forward(x, return_attention=True)
                except Exception as e:
                    print(f"Error capturing attention: {e}")
                    # Fall back to regular forward without attention
                    policy, value = model._original_forward(x, return_attention=False)
                    return policy, value, []  # Return empty attention list
            else:
                return model._original_forward(x, return_attention=False)

        # Replace the forward method
        model.forward = forward_with_attention

    return model


def restore_model_forward(model):
    """Restore the original forward method if it was modified."""
    if hasattr(model, '_original_forward'):
        model.forward = model._original_forward
        delattr(model, '_original_forward')

    return model

def get_move_with_attention(model, state, mcts_simulations=0, temperature=0.0):
    """Get a move and attention maps from the model."""
    # Create input tensor
    state_tensor = torch.tensor(state.encode_for_network(), dtype=torch.float32).unsqueeze(0)
    device = next(model.parameters()).device
    state_tensor = state_tensor.to(device)

    # Get model output with attention weights
    with torch.no_grad():
        output = model(state_tensor, return_attention=True)
        # Check how many values are returned
        if isinstance(output, tuple) and len(output) == 3:
            policy_logits, value, attention_weights = output
        else:
            print(
                f"Warning: Model returned unexpected output format: {type(output)}, length: {len(output) if isinstance(output, tuple) else 'not a tuple'}")
            # Fall back to standard processing without attention
            policy_logits, value = model(state_tensor, return_attention=False)
            attention_weights = []

    # Convert attention weights to numpy arrays for visualization
    attention_maps = []
    for layer_attn in attention_weights:
        # Average across attention heads if there are multiple heads
        if len(layer_attn.shape) > 3:  # [batch, heads, seq_len, seq_len]
            layer_attn = layer_attn.mean(dim=1)  # Average across heads

        # Convert to numpy
        layer_attn = layer_attn.squeeze(0).cpu().numpy()
        attention_maps.append(layer_attn)

    # Use MCTS if simulations > 0, otherwise use direct policy
    if mcts_simulations > 0:
        from ai.mcts import mcts_search

        # We need to temporarily restore the original forward method for MCTS
        original_forward = model.forward
        model.forward = lambda x: original_forward(x, return_attention=False)

        try:
            move, _ = mcts_search(model, state, num_simulations=mcts_simulations, temperature=temperature)
        except Exception as e:
            print(f"Error in MCTS search: {e}")
            move = None
        finally:
            # Restore the attention-enabled forward method
            model.forward = original_forward

        if move is None:
            # Fall back to direct policy
            print("MCTS failed, falling back to direct policy")
            valid_moves = state.get_valid_moves()
            if valid_moves:
                move = valid_moves[0]  # Just pick the first valid move
    else:
        # Select move based on policy (deterministic)
        from ai.training import action_to_index

        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        valid_moves = state.get_valid_moves()
        valid_indices = [action_to_index(move) for move in valid_moves]

        # Filter policy for valid moves
        valid_policy = np.zeros_like(policy)
        for idx in valid_indices:
            if idx < len(valid_policy):
                valid_policy[idx] = policy[idx]

        if valid_policy.sum() > 0:
            # Choose highest probability move
            chosen_idx = np.argmax(valid_policy)
            move = None
            for i, idx in enumerate(valid_indices):
                if idx == chosen_idx:
                    move = valid_moves[i]
                    break
        else:
            move = None

        if move is None and valid_moves:
            move = valid_moves[0]  # Fallback

    return move, value.item(), attention_maps


def visualize_attention_maps(state, attention_maps, layer_idx=None, save_path=None, show=True):
    """
    Visualize attention maps from transformer layers.

    Args:
        state: Game state
        attention_maps: List of attention weight matrices
        layer_idx: Which layer to visualize (None = all layers)
        save_path: Path to save the visualization
        show: Whether to display the visualization
    """
    num_layers = len(attention_maps)

    if layer_idx is not None:
        # Visualize only the specified layer
        attention_maps = [attention_maps[layer_idx]]
        layers_to_show = [layer_idx]
    else:
        # Visualize all layers
        layers_to_show = range(num_layers)

    # Create figure with subplots for each layer
    fig, axes = plt.subplots(len(layers_to_show), 1, figsize=(12, 5 * len(layers_to_show)))
    if len(layers_to_show) == 1:
        axes = [axes]  # Make it iterable

    for i, (layer_num, attn_map) in enumerate(zip(layers_to_show, attention_maps)):
        ax = axes[i]

        # Reshape attention map to board dimensions
        # Attention is typically [seq_len, seq_len], where seq_len is board_size^2
        board_size = BOARD_SIZE
        attn_size = attn_map.shape[0]

        # Check if attention map is the expected size
        if attn_size != board_size * board_size:
            print(f"Warning: Attention map size {attn_size} doesn't match board size {board_size}^2")
            # Try to handle different sizes by taking a subset or duplicating
            if attn_size > board_size * board_size:
                attn_map = attn_map[:board_size * board_size, :board_size * board_size]
            else:
                # Create empty larger map and fill with available data
                temp_map = np.zeros((board_size * board_size, board_size * board_size))
                temp_map[:attn_map.shape[0], :attn_map.shape[1]] = attn_map
                attn_map = temp_map

        # For visualization, let's look at the average attention given to each position
        avg_attention = attn_map.mean(axis=0).reshape(board_size, board_size)

        # Plot as heatmap
        im = ax.imshow(avg_attention, cmap='viridis')
        ax.set_title(f'Layer {layer_num + 1}: Average Attention per Position')

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Draw grid
        for x in range(board_size + 1):
            ax.axhline(x - 0.5, color='black', linewidth=0.5)
            ax.axvline(x - 0.5, color='black', linewidth=0.5)

        # Label axes
        ax.set_xticks(range(board_size))
        ax.set_yticks(range(board_size))
        ax.set_xticklabels(range(board_size))
        ax.set_yticklabels(range(board_size))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')

        # Add board pieces as annotations
        for y in range(board_size):
            for x in range(board_size):
                piece = state.board[y][x]
                if piece != 0:  # Not empty
                    # Determine piece symbol and color based on numeric value
                    if piece == TileType.FIRE.value:
                        symbol = 'F'
                        color = 'red'
                    elif piece == TileType.WATER.value:
                        symbol = 'W'
                        color = 'blue'
                    elif piece == TileType.AIR.value:
                        symbol = 'A'
                        color = 'white'
                    elif piece == TileType.EARTH.value:
                        symbol = 'E'
                        color = 'brown'
                    else:
                        symbol = str(piece)
                        color = 'gray'

                    # Add piece symbol to the plot
                    text_color = 'black' if color in ['white', 'yellow'] else 'white'
                    ax.text(x, y, symbol, ha='center', va='center', color=text_color,
                            fontweight='bold', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved attention visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_attention_for_move(model, state, attention_maps, move, value, output_dir, show=True):
    """
    Create a comprehensive visualization of attention for a specific move.

    Args:
        model: The neural network model
        state: Current game state
        attention_maps: List of attention weight matrices
        move: The move that was made
        value: Model's evaluation of the position
        output_dir: Directory to save visualizations
        show: Whether to display visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Create a figure for the current board state and move
    plt.figure(figsize=(14, 10))

    # Subplot 1: Visualize the board
    plt.subplot(2, 2, 1)

    # Create a basic visualization of the board
    board_img = np.zeros((BOARD_SIZE, BOARD_SIZE, 3))

    # Fill in pieces
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            piece = state.board[y][x]
            # Check if the piece is not empty (assuming 0 is empty)
            if piece != 0:
                # Set color based on piece value
                # Adjust these comparisons based on your actual TileType enum values
                if piece == TileType.FIRE.value:
                    board_img[y, x] = [1, 0, 0]  # Red
                elif piece == TileType.WATER.value:
                    board_img[y, x] = [0, 0, 1]  # Blue
                elif piece == TileType.AIR.value:
                    board_img[y, x] = [1, 1, 1]  # White
                elif piece == TileType.EARTH.value:
                    board_img[y, x] = [0.5, 0.25, 0]  # Brown
                else:
                    board_img[y, x] = [0.5, 0.5, 0.5]  # Gray

    plt.imshow(board_img)

    # Draw grid
    for i in range(BOARD_SIZE + 1):
        plt.axhline(i - 0.5, color='black', linewidth=0.5)
        plt.axvline(i - 0.5, color='black', linewidth=0.5)

    # Add coordinates
    for i in range(BOARD_SIZE):
        plt.text(-0.5, i, str(i), ha='center', va='center')
        plt.text(i, -0.5, str(i), ha='center', va='center')

    plt.title(f"Current Board (Player {state.current_player}'s turn)")

    # Mark the move on the board if it's a move action
    if move and move[0] == "move":
        _, from_x, from_y, to_x, to_y = move
        plt.plot(from_x, from_y, 'o', color='yellow', markersize=10, alpha=0.7)
        plt.arrow(from_x, from_y, to_x - from_x, to_y - from_y,
                  color='yellow', width=0.1, head_width=0.3, alpha=0.7)
    elif move and move[0] == "plant":
        _, tile_type, x, y = move
        plt.plot(x, y, '*', color='yellow', markersize=12, alpha=0.7)

    # Subplot 2: First layer attention map
    plt.subplot(2, 2, 2)
    if attention_maps and len(attention_maps) > 0:
        # Use the first layer's attention
        attn_map = attention_maps[0]

        # Calculate average attention per position
        avg_attention = attn_map.mean(axis=0).reshape(BOARD_SIZE, BOARD_SIZE)

        plt.imshow(avg_attention, cmap='viridis')
        plt.title("Layer 1: Average Attention")
        plt.colorbar()

        # Draw grid
        for i in range(BOARD_SIZE + 1):
            plt.axhline(i - 0.5, color='black', linewidth=0.5)
            plt.axvline(i - 0.5, color='black', linewidth=0.5)
    else:
        plt.text(0.5, 0.5, "No attention data available",
                 ha='center', va='center', transform=plt.gca().transAxes)

    # Subplot 3: Last layer attention map
    plt.subplot(2, 2, 3)
    if attention_maps and len(attention_maps) > 1:
        # Use the last layer's attention
        attn_map = attention_maps[-1]

        # Calculate average attention per position
        avg_attention = attn_map.mean(axis=0).reshape(BOARD_SIZE, BOARD_SIZE)

        plt.imshow(avg_attention, cmap='viridis')
        plt.title(f"Layer {len(attention_maps)}: Average Attention")
        plt.colorbar()

        # Draw grid
        for i in range(BOARD_SIZE + 1):
            plt.axhline(i - 0.5, color='black', linewidth=0.5)
            plt.axvline(i - 0.5, color='black', linewidth=0.5)
    else:
        plt.text(0.5, 0.5, "No last layer attention data available",
                 ha='center', va='center', transform=plt.gca().transAxes)

    # Subplot 4: Information and stats
    plt.subplot(2, 2, 4)
    plt.axis('off')

    # Add textual information
    info_text = f"Turn: {state.turn_number}\n"
    info_text += f"Player: {state.current_player}\n"
    info_text += f"Move: {move}\n"
    info_text += f"Position Value: {value:.3f}\n\n"

    # Add model information
    if hasattr(model, 'conv1'):
        info_text += f"Model Architecture:\n"
        info_text += f"- Input Channels: {model.conv1.in_channels}\n"
        info_text += f"- Model Dimension: {model.d_model}\n"
        info_text += f"- Attention Heads: {model.nhead}\n"
        info_text += f"- Layers: {model.num_layers}\n"

    plt.text(0.1, 0.9, info_text, ha='left', va='top', transform=plt.gca().transAxes)

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(output_dir, f"move_attention_{timestamp}.png")
    plt.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    # Also save individual layer attention maps
    for i, attn_map in enumerate(attention_maps):
        layer_save_path = os.path.join(output_dir, f"layer_{i + 1}_attention_{timestamp}.png")
        visualize_attention_maps(state, [attn_map], 0, save_path=layer_save_path, show=False)

    return save_path


def safe_mcts_search(model, state, num_simulations=800, temperature=1.0):
    """
    A safer version of MCTS search that handles models that might have been modified
    for attention visualization.

    Args:
        model: Neural network model
        state: Current game state
        num_simulations: Number of MCTS simulations to run
        temperature: Temperature for final move selection

    Returns:
        Best move according to search, policy probabilities for training
    """
    from ai.mcts import mcts_search

    # Check if model has been modified for attention
    has_attention_mod = hasattr(model, '_original_forward')

    if has_attention_mod:
        # Temporarily restore original forward method
        original_forward = model.forward
        model.forward = model._original_forward

    try:
        # Run normal MCTS search
        move, policy = mcts_search(model, state, num_simulations=num_simulations, temperature=temperature)
    except Exception as e:
        print(f"Error in MCTS search: {str(e)}")
        # Fall back to direct policy
        state_tensor = torch.tensor(state.encode_for_network(), dtype=torch.float32).unsqueeze(0)
        device = next(model.parameters()).device
        state_tensor = state_tensor.to(device)

        with torch.no_grad():
            policy_logits, _ = model(state_tensor)

        policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        # Just pick the first valid move as fallback
        valid_moves = state.get_valid_moves()
        move = valid_moves[0] if valid_moves else None
    finally:
        # Restore attention-enabled forward method if it was modified
        if has_attention_mod:
            model.forward = original_forward

    return move, policy


def main():
    parser = argparse.ArgumentParser(description='Visualize Skud Pai Sho AI attention patterns')

    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file')

    # Visualization mode
    parser.add_argument('--mode', type=str, default='game', choices=['game', 'position'],
                        help='Visualization mode: game or position')

    # Game mode parameters
    parser.add_argument('--num_moves', type=int, default=10,
                        help='Number of moves to play and visualize')
    parser.add_argument('--mcts_sims', type=int, default=400,
                        help='Number of MCTS simulations per move')
    parser.add_argument('--player_side', type=int, default=2, choices=[1, 2],
                        help='Which player you want to play as (1 or 2)')

    # Position mode parameters
    parser.add_argument('--moves', type=str, default=None,
                        help='Comma-separated list of moves to reach position')
    parser.add_argument('--game_file', type=str, default=None,
                        help='Game history file to load')
    parser.add_argument('--move_number', type=int, default=-1,
                        help='Move number to analyze from game file')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='attention_visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not display visualizations interactively')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Create sample state to determine input channels
    sample_state = SkudPaiShoState()
    encoded_state = sample_state.encode_for_network()
    input_channels = encoded_state.shape[0]

    # Get model architecture from checkpoint or use default
    d_model = checkpoint.get('d_model', 256)
    nhead = checkpoint.get('nhead', 8)
    num_layers = checkpoint.get('num_layers', 3)
    dropout = checkpoint.get('dropout', 0.0)  # Use 0 dropout for visualization

    model = SkudPaiShoTransformer(
        input_channels=input_channels,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    model = model.to(device)

    # Modify model to return attention weights
    model = modify_model_for_attention(model)

    if args.mode == 'game':
        # Initialize the game
        state = SkudPaiShoState()

        print("\n=== Playing and Visualizing AI Attention ===")
        print(f"Model will play {args.num_moves} moves")

        for move_num in range(args.num_moves):
            print(f"\nMove {move_num + 1}/{args.num_moves}")
            print(f"Current player: {state.current_player}")

            if state.current_player != args.player_side:
                # AI's turn
                print("AI is thinking...")
                move, value, attention_maps = get_move_with_attention(
                    model,
                    state,
                    mcts_simulations=args.mcts_sims,
                    temperature=0.0
                )

                print(f"AI plays: {move} (value: {value:.3f})")

                # Visualize attention for this move
                vis_path = visualize_attention_for_move(
                    model,
                    state,
                    attention_maps,
                    move,
                    value,
                    args.output_dir,
                    not args.no_show
                )

                print(f"Saved visualization to {vis_path}")

                # Make the move
                state.make_move(move)
            else:
                # Human player's turn
                valid_moves = state.get_valid_moves()

                print("Valid moves:")
                for i, move in enumerate(valid_moves):
                    print(f"{i + 1}: {move}")

                # Auto-select first move for human for demonstration purposes
                selected_move = valid_moves[0]
                print(f"Auto-selecting move: {selected_move}")

                # Make the move
                state.make_move(selected_move)

            if state.is_game_over():
                print("\nGame over!")
                result = state.get_reward(args.player_side)
                if result > 0:
                    print("You win!")
                elif result < 0:
                    print("AI wins!")
                else:
                    print("It's a draw!")
                break

    elif args.mode == 'position':
        # Get state from moves or game file
        state = None
        if args.moves:
            # Create position from move sequence
            moves = [m.strip() for m in args.moves.split(',')]
            state = create_position_from_moves(moves)

        elif args.game_file:
            # Load game history
            with open(args.game_file, 'r') as f:
                game_data = json.load(f)

            # Create position from game
            move_list = []
            for move_data in game_data.get('moves', []):
                if args.move_number < 0 or move_data.get('turn', 0) <= args.move_number:
                    move_str = move_data.get('move', '')
                    if move_str and move_str != "Unknown":
                        move_list.append(move_str)

            state = create_position_from_moves(move_list)

        if state is None:
            print("Failed to create valid position. Please check your inputs.")
            return

        # Get attention for this position
        move, value, attention_maps = get_move_with_attention(
            model,
            state,
            mcts_simulations=args.mcts_sims,
            temperature=0.0
        )

        print(f"AI would play: {move} (value: {value:.3f})")

        # Visualize attention for all layers
        for i, attn_map in enumerate(attention_maps):
            layer_save_path = os.path.join(args.output_dir, f"position_layer_{i + 1}_attention.png")
            visualize_attention_maps(
                state,
                [attn_map],
                0,
                save_path=layer_save_path,
                show=not args.no_show
            )
            print(f"Saved layer {i + 1} attention to {layer_save_path}")

        # Also create comprehensive visualization
        vis_path = visualize_attention_for_move(
            model,
            state,
            attention_maps,
            move,
            value,
            args.output_dir,
            not args.no_show
        )

        print(f"Saved comprehensive visualization to {vis_path}")


if __name__ == "__main__":
    # Import action_to_index (should be in your utils.py or training.py)
    from ai.training import action_to_index


    # Import function to create position from moves (defined elsewhere in this script)
    def create_position_from_moves(moves):
        """Create a game state by applying a sequence of moves."""
        state = SkudPaiShoState()

        for move in moves:
            if isinstance(move, str):
                # Parse string representation of move
                parts = move.strip().replace("(", "").replace(")", "").replace("'", "").split(",")
                if parts[0].strip() == "plant":
                    tile_type = getattr(TileType, parts[1].strip())
                    x, y = int(parts[2].strip()), int(parts[3].strip())
                    move = ("plant", tile_type, x, y)
                elif parts[0].strip() == "move":
                    from_x = int(parts[1].strip())
                    from_y = int(parts[2].strip())
                    to_x = int(parts[3].strip())
                    to_y = int(parts[4].strip())
                    move = ("move", from_x, from_y, to_x, to_y)

            state.make_move(move)

        return state


    main()