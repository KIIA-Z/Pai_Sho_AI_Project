#!/usr/bin/env python
# evaluate.py - Script for evaluating Skud Pai Sho AI performance

import os
import argparse
import torch
import numpy as np
import random
import time
import json
import matplotlib.pyplot as plt

from game.state import SkudPaiShoState
from ai.model import SkudPaiShoTransformer
from ai.utils import evaluate_model
from ai.opening_book import OpeningBook


def main():
    parser = argparse.ArgumentParser(description='Evaluate Skud Pai Sho AI model')
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file')
    parser.add_argument('--opponent_model', type=str, default=None, help='Path to opponent model file')

    # Evaluation parameters
    parser.add_argument('--num_games', type=int, default=20, help='Number of games to play')
    parser.add_argument('--mcts_sims', type=int, default=100, help='Number of MCTS simulations per move')
    parser.add_argument('--opponent_type', type=str, default='model',
                        choices=['random', 'mcts', 'model'],
                        help='Type of opponent')

    # Opening book parameters
    parser.add_argument('--use_opening_book', action='store_true', help='Use opening book')
    parser.add_argument('--opening_book_file', type=str, default='data/opening_book.json',
                        help='Opening book file')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--verbose', action='store_true', help='Print detailed evaluation progress')
    parser.add_argument('--save_games', action='store_true', help='Save game histories')

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
    dropout = checkpoint.get('dropout', 0.0)  # Use 0 dropout for evaluation

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

    # Load opponent model if specified
    opponent = None
    if args.opponent_type == 'model' and args.opponent_model:
        print(f"Loading opponent model from {args.opponent_model}")
        opponent_checkpoint = torch.load(args.opponent_model, map_location=device)

        opponent = SkudPaiShoTransformer(
            input_channels=input_channels,
            d_model=d_model,  # Use same architecture for simplicity
            nhead=nhead,
            num_layers=num_layers,
            dropout=0.0
        )
        opponent.load_state_dict(opponent_checkpoint['model_state_dict'])
        opponent.eval()
        opponent = opponent.to(device)
    elif args.opponent_type == 'mcts':
        opponent = "mcts"
    else:
        opponent = None  # Random opponent

    # Load opening book if specified
    opening_book = None
    if args.use_opening_book and os.path.exists(args.opening_book_file):
        opening_book = OpeningBook(book_file=args.opening_book_file)
        print(f"Loaded opening book with {opening_book.get_stats()['total_positions']} positions")

    # Run evaluation
    print(f"Starting evaluation: {args.num_games} games against {args.opponent_type} opponent")
    start_time = time.time()

    results = evaluate_model(
        model,
        num_games=args.num_games,
        opponent=opponent,
        opening_book=opening_book,
        mcts_simulations=args.mcts_sims,
        verbose=args.verbose
    )

    elapsed_time = time.time() - start_time

    # Add timing information to results
    results['eval_time'] = elapsed_time
    results['avg_time_per_game'] = elapsed_time / args.num_games
    results['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
    results['model_path'] = args.model_path
    results['opponent_type'] = args.opponent_type
    results['opponent_model'] = args.opponent_model
    results['mcts_simulations'] = args.mcts_sims

    # Print summary
    print("\nEvaluation Results:")
    print(f"Win Rate: {results['win_rate']:.3f}")
    print(f"Loss Rate: {results['loss_rate']:.3f}")
    print(f"Draw Rate: {results['draw_rate']:.3f}")
    print(f"Average Game Length: {results['avg_game_length']:.1f} moves")
    print(f"Average Harmony Count: {results['avg_harmony_count']:.2f}")
    print(f"Evaluation Time: {elapsed_time:.1f}s ({results['avg_time_per_game']:.1f}s per game)")

    # Save results to file
    results_file = os.path.join(args.output_dir, f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {results_file}")

    # Create visualization
    plt.figure(figsize=(10, 6))

    # Plot win/loss/draw rates
    labels = ['Win', 'Loss', 'Draw']
    rates = [results['win_rate'], results['loss_rate'], results['draw_rate']]
    colors = ['green', 'red', 'gray']

    plt.bar(labels, rates, color=colors)
    plt.ylabel('Rate')
    plt.title(f'Evaluation Results (vs {args.opponent_type})')
    plt.ylim(0, 1)

    for i, v in enumerate(rates):
        plt.text(i, v + 0.05, f"{v:.3f}", ha='center')

    # Save figure
    plt.savefig(os.path.join(args.output_dir, f"eval_results_{time.strftime('%Y%m%d_%H%M%S')}.png"))
    plt.close()


if __name__ == "__main__":
    main()