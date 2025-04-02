#!/usr/bin/env python
# analyze.py - Script for analyzing Skud Pai Sho positions

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import time

from game.state import SkudPaiShoState, TileType, BOARD_SIZE
from ai.model import SkudPaiShoTransformer
from ai.utils import analyze_position
from ai.mcts import mcts_search


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


def visualize_analysis(state, top_moves, value, output_path=None):
    """Visualize position analysis with top moves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot board position (simplified)
    ax1.set_title(f"Position (Player {state.current_player}'s turn)")

    # Create a basic board representation
    # This is a placeholder - you should adapt it to match your actual game representation
    board_img = np.zeros((BOARD_SIZE, BOARD_SIZE, 3))

    # Example: Mark cells with pieces
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            piece = state.board[y][x]
            if piece.type != TileType.EMPTY:
                # Set color based on piece type
                if piece.type == TileType.FIRE:
                    board_img[y, x] = [1, 0, 0]  # Red
                elif piece.type == TileType.WATER:
                    board_img[y, x] = [0, 0, 1]  # Blue
                elif piece.type == TileType.AIR:
                    board_img[y, x] = [1, 1, 1]  # White
                elif piece.type == TileType.EARTH:
                    board_img[y, x] = [0.5, 0.25, 0]  # Brown
                else:
                    board_img[y, x] = [0.5, 0.5, 0.5]  # Gray

    ax1.imshow(board_img)

    # Draw grid
    for i in range(BOARD_SIZE + 1):
        ax1.axhline(i - 0.5, color='black', linewidth=0.5)
        ax1.axvline(i - 0.5, color='black', linewidth=0.5)

    # Add coordinates
    for i in range(BOARD_SIZE):
        ax1.text(-0.5, i, str(i), ha='center', va='center')
        ax1.text(i, -0.5, str(i), ha='center', va='center')

    # Plot top moves
    move_probs = [prob for _, prob in top_moves]
    move_labels = [str(move) for move, _ in top_moves]

    bars = ax2.barh(range(len(move_probs)), move_probs, color='skyblue')
    ax2.set_yticks(range(len(move_probs)))
    ax2.set_yticklabels(move_labels)
    ax2.set_xlabel('Probability')
    ax2.set_title(f'Top Moves (Value: {value:.3f})')

    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, move_probs)):
        ax2.text(max(prob + 0.01, 0.05), i, f'{prob:.1%}', va='center')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Analysis visualization saved to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze Skud Pai Sho positions')
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file')

    # Analysis parameters
    parser.add_argument('--use_mcts', action='store_true', help='Use MCTS for deeper analysis')
    parser.add_argument('--mcts_sims', type=int, default=1600, help='Number of MCTS simulations')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top moves to show')

    # Position specification (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--fen', type=str, help='Position in FEN-like notation')
    group.add_argument('--moves', type=str, help='Comma-separated list of moves to reach position')
    group.add_argument('--game_file', type=str, help='Game history file to load')
    group.add_argument('--move_number', type=int, default=-1, help='Move number to analyze from game file')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='Directory to save analysis results')
    parser.add_argument('--save_visualization', action='store_true', help='Save analysis visualization')

    args = parser.parse_args()

    # Create output directory if saving visualization
    if args.save_visualization:
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
    dropout = checkpoint.get('dropout', 0.0)  # Use 0 dropout for analysis

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

    # Get position to analyze
    state = None

    if args.fen:
        # Parse FEN-like notation
        # This is a placeholder - you need to implement this based on your game
        print("FEN parsing not implemented - please use --moves or --game_file")
        return

    elif args.moves:
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

    # Analyze position
    print("\nAnalyzing position...")
    print(f"Current player: {state.current_player}")
    print(f"Turn number: {state.turn_number}")

    # Perform analysis
    start_time = time.time()
    value, top_moves = analyze_position(
        model,
        state,
        top_k=args.top_k,
        use_mcts=args.use_mcts,
        mcts_simulations=args.mcts_sims
    )
    analysis_time = time.time() - start_time

    # Print results
    print(f"\nPosition evaluation: {value:.3f}")
    print(f"Top {len(top_moves)} moves:")
    for i, (move, prob) in enumerate(top_moves):
        print(f"{i + 1}. {move} - {prob:.2%}")

    print(f"\nAnalysis completed in {analysis_time:.2f} seconds")

    # Create visualization
    if args.save_visualization:
        output_path = os.path.join(
            args.output_dir,
            f"analysis_{time.strftime('%Y%m%d_%H%M%S')}.png"
        )
    else:
        output_path = None

    visualize_analysis(state, top_moves, value, output_path)


if __name__ == "__main__":
    main()