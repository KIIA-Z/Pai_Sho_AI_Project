import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt  # Add this missing import
from game.state import SkudPaiShoState
from ai.model import SkudPaiShoTransformer
from ai.utils import create_initial_model
from ai.integrated_training import train_skud_pai_sho_ai_with_mcts
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import json
from ai.opening_book import OpeningBook


def plot_training_metrics(metrics, save_path):
    """Plot and save training metrics."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # Policy Loss
    axes[0, 0].plot(metrics["iterations"], metrics["policy_losses"])
    axes[0, 0].set_title("Policy Loss vs. Iterations")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Policy Loss")
    axes[0, 0].grid(True)

    # Value Loss
    axes[0, 1].plot(metrics["iterations"], metrics["value_losses"])
    axes[0, 1].set_title("Value Loss vs. Iterations")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("Value Loss")
    axes[0, 1].grid(True)

    # Win Rate
    axes[1, 0].plot(metrics["iterations"], metrics["win_rates"])
    axes[1, 0].set_title("Win Rate vs. Iterations")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Win Rate")
    axes[1, 0].grid(True)

    # Game Length
    axes[1, 1].plot(metrics["iterations"], metrics["game_lengths"])
    axes[1, 1].set_title("Average Game Length vs. Iterations")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Game Length (moves)")
    axes[1, 1].grid(True)

    # Harmony Count
    axes[2, 0].plot(metrics["iterations"], metrics["harmony_counts"])
    axes[2, 0].set_title("Average Harmony Count vs. Iterations")
    axes[2, 0].set_xlabel("Iteration")
    axes[2, 0].set_ylabel("Harmony Count")
    axes[2, 0].grid(True)

    # Learning Rate
    axes[2, 1].plot(metrics["iterations"], metrics["learning_rates"])
    axes[2, 1].set_title("Learning Rate vs. Iterations")
    axes[2, 1].set_xlabel("Iteration")
    axes[2, 1].set_ylabel("Learning Rate")
    axes[2, 1].set_yscale('log')
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Skud Pai Sho AI with improved MCTS')
    parser.add_argument('--iterations', type=int, default=50, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=20, help='Number of self-play games per iteration')
    parser.add_argument('--mcts_sims', type=int, default=800, help='Number of MCTS simulations per move')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs per iteration')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load initial model')
    parser.add_argument('--architecture', type=str, default='A1', help='Model architecture (A1-A5)')
    parser.add_argument('--use_opening_book', action='store_true', help='Use opening book')
    parser.add_argument('--opening_book_file', type=str, default='data/opening_book.json', help='Opening book file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output and reduced settings')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Debug mode - reduce settings for faster execution
    if args.debug:
        print("DEBUG MODE: Using reduced settings for faster execution")
        args.mcts_sims = 5  # Very few simulations
        args.games = 2      # Just 2 games per iteration
        args.iterations = 2 # Just 2 iterations
        args.epochs = 2     # Just 2 epochs

    # Create sample state to determine input channels
    sample_state = SkudPaiShoState()
    encoded_state = sample_state.encode_for_network()
    input_channels = encoded_state.shape[0]

    print(f"Detected input channels from state encoding: {input_channels}")

    # Create or load model
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)

        # Get model architecture from checkpoint or use default
        d_model = checkpoint.get('d_model', 256)
        nhead = checkpoint.get('nhead', 8)
        num_layers = checkpoint.get('num_layers', 3)
        dropout = checkpoint.get('dropout', 0.2)

        model = SkudPaiShoTransformer(
            input_channels=input_channels,  # Use detected input channels
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Creating new model with architecture {args.architecture}")
        model = create_initial_model(
            input_channels=input_channels,  # Use detected input channels
            architecture=args.architecture,
            device=None  # We'll move to device later
        )

    # Move model to device
    model = model.to(device)

    # Print model summary
    print(f"Model structure:")
    print(f"  Input channels: {input_channels}")
    print(f"  First conv layer expects: {model.conv1.in_channels} channels")

    # Train model
    model, training_log = train_skud_pai_sho_ai_with_mcts(
        model=model,
        iterations=args.iterations,
        games_per_iteration=args.games,
        mcts_simulations=args.mcts_sims,
        epochs_per_iteration=args.epochs,
        use_opening_book=args.use_opening_book,
        opening_book_file=args.opening_book_file,
        model_dir=args.model_dir,
        log_dir=args.log_dir
    )

    # If we have training data, plot it
    if training_log and len(training_log["iterations"]) > 0:
        print("Plotting training metrics...")
        plot_path = os.path.join(args.log_dir, "training_metrics.png")
        plot_training_metrics(training_log, plot_path)
        print(f"Plots saved to {plot_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()