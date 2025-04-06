#!/usr/bin/env python
# evaluate_skud_pai_sho_enhanced.py - Evaluation script for both transformer and Q-learning models

import os
import argparse
import torch
import pickle
import time
import json
from tqdm import tqdm

from game.state import SkudPaiShoState
from ai.model import SkudPaiShoTransformer
from ai.opening_book import OpeningBook
from q_learning.q_learning_ai import SkudPaiShoQLearning

# Import the enhanced evaluation framework
from dual_model_framework import SkudPaiShoEvaluationEnhanced

def main():
    parser = argparse.ArgumentParser(description='Enhanced Skud Pai Sho AI Evaluation Framework')
    
    # Model parameters
    parser.add_argument('--transformer_models', type=str, nargs='+', default=[], 
                      help='Paths to transformer model files')
    parser.add_argument('--q_learning_models', type=str, nargs='+', default=[], 
                      help='Paths to Q-learning model files')
    parser.add_argument('--model_names', type=str, nargs='+', default=[], 
                      help='Names for the models (optional, in order of transformer then q_learning models)')
    
    # Evaluation parameters
    parser.add_argument('--num_games', type=int, default=20, 
                      help='Number of games to generate for each matchup')
    parser.add_argument('--mcts_sims', type=int, default=100, 
                      help='Number of MCTS simulations per move (for transformer models)')
    parser.add_argument('--use_games_from', type=str, default=None,
                      help='Load previously played games from file instead of generating new ones')
    
    # Opening book parameters
    parser.add_argument('--use_opening_book', action='store_true', 
                      help='Use opening book')
    parser.add_argument('--opening_book_file', type=str, default='data/opening_book.json',
                      help='Opening book file')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--verbose', action='store_true', 
                      help='Print detailed evaluation progress')
    
    args = parser.parse_args()
    
    # Ensure we have at least one model of each type
    if not args.transformer_models and not args.q_learning_models:
        print("Error: You must specify at least one model using --transformer_models or --q_learning_models")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize enhanced evaluation framework
    framework = SkudPaiShoEvaluationEnhanced(output_dir=args.output_dir)
    
    # Process model names
    model_names = args.model_names.copy() if args.model_names else []
    
    # Load transformer models
    for i, model_path in enumerate(args.transformer_models):
        # Get model name
        if i < len(model_names):
            model_name = model_names[i]
        else:
            model_name = f"transformer_{i+1}"
        
        # Add model
        framework.add_model(model_name, model_path, model_type="transformer")
    
    # Load Q-learning models
    for i, model_path in enumerate(args.q_learning_models):
        # Get model name
        if i + len(args.transformer_models) < len(model_names):
            model_name = model_names[i + len(args.transformer_models)]
        else:
            model_name = f"q_learning_{i+1}"
        
        # Add model
        framework.add_model(model_name, model_path, model_type="q_learning")
    
    # Load opening book if specified
    opening_book = None
    if args.use_opening_book:
        if os.path.exists(args.opening_book_file):
            print(f"Loading opening book from {args.opening_book_file}")
            opening_book = OpeningBook(book_file=args.opening_book_file)
        else:
            print(f"Opening book file {args.opening_book_file} not found. Continuing without it.")
    
    # Load existing games or generate new ones
    if args.use_games_from:
        if os.path.exists(args.use_games_from):
            num_games = framework.load_game_data(args.use_games_from)
            print(f"Loaded {num_games} games from {args.use_games_from}")
        else:
            print(f"Game data file {args.use_games_from} not found. Generating new games.")
            
            # Create all possible matchups
            matchups = []
            model_names = list(framework.models.keys())
            if len(model_names) >= 2:
                for i, name1 in enumerate(model_names):
                    for j, name2 in enumerate(model_names):
                        if i != j:  # Avoid self-play
                            matchups.append((name1, name2))
            else:
                print("Error: Need at least 2 models to generate games")
                return
            
            # Generate games
            num_games = framework.generate_games(
                num_games=args.num_games,
                matchups=matchups,
                mcts_simulations=args.mcts_sims,
                opening_book=opening_book,
                verbose=args.verbose
            )
            
            # Save the generated games
            game_data_file = os.path.join(args.output_dir, "game_data.json")
            framework.save_game_data(game_data_file)
            print(f"Generated and saved {num_games} games to {game_data_file}")
    else:
        # Create all possible matchups
        matchups = []
        model_names = list(framework.models.keys())
        if len(model_names) >= 2:
            for i, name1 in enumerate(model_names):
                for j, name2 in enumerate(model_names):
                    if i != j:  # Avoid self-play
                        matchups.append((name1, name2))
        else:
            print("Error: Need at least 2 models to generate games")
            return
        
        # Generate games
        num_games = framework.generate_games(
            num_games=args.num_games,
            matchups=matchups,
            mcts_simulations=args.mcts_sims,
            opening_book=opening_book,
            verbose=args.verbose
        )
        
        # Save the generated games
        game_data_file = os.path.join(args.output_dir, "game_data.json")
        framework.save_game_data(game_data_file)
        print(f"Generated and saved {num_games} games to {game_data_file}")
    
    # Run the evaluation
    print("\nRunning enhanced evaluation analyses...")
    start_time = time.time()
    
    results = framework.run_functional_evaluation()
    
    elapsed_time = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed_time:.1f} seconds!")
    print(f"Results saved to {args.output_dir}")
    print(f"HTML report available at {os.path.join(args.output_dir, 'evaluation_report.html')}")

    # Print summary of results
    if 'win_rates' in results:
        print("\nWin Rate Summary:")
        for model, stats in results['win_rates'].items():
            model_type = stats.get('model_type', 'unknown')
            print(f"{model} ({model_type}): {stats['win_rate']:.3f} win rate ({stats['games_played']} games)")
    
    if 'computational_efficiency' in results:
        print("\nComputational Efficiency Summary:")
        for model, stats in results['computational_efficiency'].items():
            model_type = stats.get('model_type', 'unknown')
            print(f"{model} ({model_type}): {stats['avg_time']*1000:.2f}ms avg thinking time")

if __name__ == "__main__":
    main()