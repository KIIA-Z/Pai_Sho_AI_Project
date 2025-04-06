#!/usr/bin/env python
# skud_pai_sho_evaluation_enhanced.py - Evaluation framework for both transformer and Q-learning models

import os
import torch
import pickle
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from game.state import SkudPaiShoState, TileType, BOARD_SIZE
from ai.model import SkudPaiShoTransformer
from ai.utils import evaluate_model
from q_learning.q_learning_ai import SkudPaiShoQLearning  # Import the Q-learning agent class


class SkudPaiShoEvaluationEnhanced:
    """Enhanced evaluation framework for Skud Pai Sho AI models (both transformer and Q-learning)."""
    
    def __init__(self, models=None, game_data=None, output_dir="evaluation_results"):
        """
        Initialize the evaluation framework.
        
        Args:
            models: Dictionary of model instances to evaluate {name: (model, type)}
            game_data: Previously played games or None to generate new games
            output_dir: Directory to save evaluation results
        """
        self.models = models or {}  # Dict of {name: (model_instance, model_type)}
        self.game_data = game_data or []
        self.output_dir = output_dir
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def add_model(self, name, model_path, model_type="transformer", **kwargs):
        """
        Add a model to the evaluation framework.
        
        Args:
            name: Name of the model
            model_path: Path to the model file
            model_type: Type of model ('transformer' or 'q_learning')
            **kwargs: Additional parameters for model initialization
        """
        print(f"Loading {model_type} model '{name}' from {model_path}")
        
        if model_type == "transformer":
            # Transformer model loading logic
            try:
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Create sample state to determine input channels
                sample_state = SkudPaiShoState()
                encoded_state = sample_state.encode_for_network()
                input_channels = encoded_state.shape[0]
                
                # Get model architecture from checkpoint if available
                d_model = kwargs.get('d_model', checkpoint.get('d_model', 256))
                nhead = kwargs.get('nhead', checkpoint.get('nhead', 8))
                num_layers = kwargs.get('num_layers', checkpoint.get('num_layers', 3))
                
                # Create model
                model = SkudPaiShoTransformer(
                    input_channels=input_channels,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dropout=0.0  # Use 0 dropout for evaluation
                )
                
                # Load weights
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()  # Set to evaluation mode
                model = model.to(self.device)
                
                # Store model and its type
                self.models[name] = (model, model_type)
                
                return model
            except Exception as e:
                print(f"Error loading transformer model: {e}")
                return None
                
        elif model_type == "q_learning":
            # Q-learning model loading logic
            try:
                # Load Q-learning model using pickle
                with open(model_path, 'rb') as f:
                    q_model = pickle.load(f)
                
                # Verify it's a SkudPaiShoQLearning instance
                if not isinstance(q_model, SkudPaiShoQLearning):
                    # If it's just the Q-table, create a new agent and set the table
                    if isinstance(q_model, dict) or isinstance(q_model, defaultdict):
                        q_table = q_model
                        q_model = SkudPaiShoQLearning()
                        q_model.q_table = defaultdict(lambda: defaultdict(float), q_table)
                    else:
                        raise TypeError("Loaded object is not a SkudPaiShoQLearning instance or Q-table")
                
                # Store model and its type
                self.models[name] = (q_model, model_type)
                
                return q_model
            except Exception as e:
                print(f"Error loading Q-learning model: {e}")
                return None
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def load_game_data(self, filepath):
        """Load previously played game data from a file."""
        with open(filepath, 'r') as f:
            self.game_data = json.load(f)
        return len(self.game_data)
    
    def save_game_data(self, filepath):
        """Save game data to a file."""
        with open(filepath, 'w') as f:
            json.dump(self.game_data, f, indent=2)
    
    def generate_games(self, num_games=20, matchups=None, mcts_simulations=100, 
                       opening_book=None, verbose=False):
        """
        Generate games between models.
        
        Args:
            num_games: Number of games to generate
            matchups: List of tuples specifying model matchups, or None for all combinations
            mcts_simulations: Number of MCTS simulations per move (for transformer models)
            opening_book: Optional opening book to use
            verbose: Print detailed progress
        """
        if len(self.models) < 2:
            raise ValueError("At least two models are required to generate games")
        
        # Default to all possible matchups if none specified
        if matchups is None:
            matchups = [(model1, model2) 
                        for i, model1 in enumerate(self.models.keys()) 
                        for model2 in list(self.models.keys())[i+1:]]
        
        all_game_data = []
        
        for model1_name, model2_name in matchups:
            print(f"Generating {num_games} games: {model1_name} vs {model2_name}")
            
            # Get models and their types
            model1, model1_type = self.models[model1_name]
            model2, model2_type = self.models[model2_name]
            
            # Track results for this matchup
            wins = {model1_name: 0, model2_name: 0, 'draw': 0}
            game_lengths = []
            harmony_counts = []
            
            for game_id in tqdm(range(num_games)):
                # Alternate who goes first
                if game_id % 2 == 0:
                    first_player = model1
                    first_player_name = model1_name
                    first_player_type = model1_type
                    second_player = model2
                    second_player_name = model2_name
                    second_player_type = model2_type
                else:
                    first_player = model2
                    first_player_name = model2_name
                    first_player_type = model2_type
                    second_player = model1
                    second_player_name = model1_name
                    second_player_type = model1_type
                
                # Play a game and record the outcome
                game_record = self._play_game(
                    first_player=first_player,
                    second_player=second_player,
                    first_player_name=first_player_name,
                    second_player_name=second_player_name,
                    first_player_type=first_player_type,
                    second_player_type=second_player_type,
                    mcts_simulations=mcts_simulations,
                    opening_book=opening_book,
                    verbose=verbose
                )
                
                # Update statistics
                winner = game_record['winner']
                if winner == 'draw':
                    wins['draw'] += 1
                else:
                    wins[winner] += 1
                
                game_lengths.append(len(game_record['moves']))
                harmony_counts.append(game_record.get('harmony_count', 0))
                
                # Add to game data
                all_game_data.append(game_record)
            
            # Print matchup results
            total_games = sum(wins.values())
            print(f"\nResults for {model1_name} vs {model2_name}:")
            print(f"{model1_name} wins: {wins[model1_name]} ({wins[model1_name]/total_games:.3f})")
            print(f"{model2_name} wins: {wins[model2_name]} ({wins[model2_name]/total_games:.3f})")
            print(f"Draws: {wins['draw']} ({wins['draw']/total_games:.3f})")
            print(f"Average game length: {sum(game_lengths)/len(game_lengths):.1f} moves")
            if harmony_counts:
                print(f"Average harmony count: {sum(harmony_counts)/len(harmony_counts):.2f}")
        
        # Add to existing game data
        self.game_data.extend(all_game_data)
        
        return len(all_game_data)
    
    def _play_game(self, first_player, second_player, first_player_name, second_player_name,
                  first_player_type, second_player_type, mcts_simulations=100, 
                  opening_book=None, verbose=False):
        """
        Play a single game between two models, adapting to model types.
        """
        from ai.mcts import mcts_search
        
        # Create a new game state
        state = SkudPaiShoState()
        
        # Initialize game record
        game_record = {
            'moves': [],
            'board_states': [],
            'thinking_times': [],
            'player_times': {first_player_name: [], second_player_name: []},
            'winner': None,
            'first_player': first_player_name,
            'second_player': second_player_name,
            'first_player_type': first_player_type,
            'second_player_type': second_player_type,
            'harmony_count': 0,
            'harmony_progression': []  # Track harmony progression throughout the game
        }
        
        # Play until game is over
        current_player = first_player
        current_player_name = first_player_name
        current_player_type = first_player_type
        
        while not state.is_game_over() and state.turn_number < 200:  # Max 200 moves
            # Record board state
            game_record['board_states'].append(state.board.tolist())
            
            # Record harmony state if available
            if hasattr(state, 'harmonies'):
                harmonies_p1 = len(state.harmonies.get(1, []))
                harmonies_p2 = len(state.harmonies.get(2, []))
                game_record['harmony_progression'].append({
                    'move': state.turn_number,
                    'player1_harmonies': harmonies_p1,
                    'player2_harmonies': harmonies_p2,
                    'total_harmonies': harmonies_p1 + harmonies_p2
                })
            
            # Get move from current player
            start_time = time.time()
            
            # Use different move selection logic based on model type
            move = None
            
            # Try using opening book first if available
            if opening_book and hasattr(opening_book, 'get_move'):
                try:
                    move = opening_book.get_move(state)
                    if verbose and move is not None:
                        print(f"Using opening book move: {move}")
                except Exception as e:
                    if verbose:
                        print(f"Error using opening book: {e}")
            
            if move is None:
                if current_player_type == "transformer":
                    # Use MCTS for transformer models
                    try:
                        move, _ = mcts_search(
                            current_player,
                            state,
                            num_simulations=mcts_simulations,
                            temperature=0.0,  # Deterministic for evaluation
                            dirichlet_noise=False
                        )
                    except Exception as e:
                        if verbose:
                            print(f"Error in MCTS: {e}")
                        # Fallback: get a random valid move
                        valid_moves = state.get_valid_moves()
                        if valid_moves:
                            move = valid_moves[0]
                        else:
                            move = None
                
                elif current_player_type == "q_learning":
                    # Use direct action selection for Q-learning agents
                    try:
                        # Set deterministic mode for evaluation (no exploration)
                        original_exploration = current_player.exploration_rate
                        current_player.exploration_rate = 0.0
                        
                        # Get move using Q-learning's choose_action method
                        move = current_player.choose_action(state)
                        
                        # Restore exploration rate
                        current_player.exploration_rate = original_exploration
                    except Exception as e:
                        if verbose:
                            print(f"Error in Q-learning action selection: {e}")
                        # Fallback: get a random valid move
                        valid_moves = state.get_valid_moves()
                        if valid_moves:
                            move = valid_moves[0]
                        else:
                            move = None
            
            thinking_time = time.time() - start_time
            
            # If no valid move, end the game
            if move is None:
                if verbose:
                    print("No valid move available. Ending game.")
                break
            
            # Record move and thinking time
            game_record['moves'].append({
                'player': current_player_name,
                'player_type': current_player_type,
                'move': str(move),  # Convert move to string for JSON serialization
                'thinking_time': thinking_time
            })
            
            game_record['thinking_times'].append(thinking_time)
            game_record['player_times'][current_player_name].append(thinking_time)
            
            # Apply move
            try:
                state.make_move(move)
            except Exception as e:
                if verbose:
                    print(f"Error making move: {e}")
                break
            
            # Record harmony count
            if hasattr(state, 'harmonies'):
                total_harmonies = sum(len(harmonies) for harmonies in state.harmonies.values())
                game_record['harmony_count'] = total_harmonies
            
            # Switch players
            if current_player == first_player:
                current_player = second_player
                current_player_name = second_player_name
                current_player_type = second_player_type
            else:
                current_player = first_player
                current_player_name = first_player_name
                current_player_type = first_player_type
        
        # Record final board state
        game_record['board_states'].append(state.board.tolist())
        
        # Record winner
        if state.is_game_over():
            winner = state.get_winner()
            if winner == 1:
                game_record['winner'] = first_player_name
            elif winner == 2:
                game_record['winner'] = second_player_name
            else:
                game_record['winner'] = 'draw'
        else:
            game_record['winner'] = 'draw'  # Draw if game ended without winner
        
        return game_record
    
    def computational_efficiency_analysis(self):
        """
        Compare time/computational efficiency between models.
        """
        print("Analyzing computational efficiency...")
        
        # Extract thinking times by model and model type
        thinking_times = {}
        for name, (model, model_type) in self.models.items():
            thinking_times[name] = {'times': [], 'type': model_type}
        
        for game in self.game_data:
            # Add thinking times for each player
            for move_data in game['moves']:
                player_name = move_data['player']
                if player_name in thinking_times:
                    thinking_times[player_name]['times'].append(move_data['thinking_time'])
        
        # Calculate statistics
        efficiency_stats = {}
        for model_name, data in thinking_times.items():
            times = data['times']
            model_type = data['type']
            
            if times:
                efficiency_stats[model_name] = {
                    'avg_time': sum(times) / len(times),
                    'median_time': np.median(times),
                    'max_time': max(times),
                    'min_time': min(times),
                    'std_time': np.std(times),
                    'total_moves': len(times),
                    'model_type': model_type
                }
        
        # Save results
        with open(os.path.join(self.output_dir, 'computational_efficiency.json'), 'w') as f:
            json.dump(efficiency_stats, f, indent=2)
        
        self.results['computational_efficiency'] = efficiency_stats
        
        # Visualization
        plt.figure(figsize=(12, 6))
        
        # Prepare data for visualization
        efficiency_df = pd.DataFrame([
            {
                'Model': model,
                'Average Time (ms)': stats['avg_time'] * 1000,  # Convert to ms
                'Median Time (ms)': stats['median_time'] * 1000,
                'Std Dev (ms)': stats['std_time'] * 1000,
                'Type': stats['model_type']
            }
            for model, stats in efficiency_stats.items()
        ])
        
        if not efficiency_df.empty:
            # Plot average times with coloring by model type
            ax = sns.barplot(x='Model', y='Average Time (ms)', hue='Type', data=efficiency_df,
                          palette={'transformer': 'blue', 'q_learning': 'green'})
            
            # Add error bars using standard deviation
            for i, row in enumerate(efficiency_df.itertuples()):
                ax.errorbar(i, row._2, yerr=row._4, fmt='none', ecolor='black', capsize=5)
            
            plt.title('Computational Efficiency: Average Thinking Time by Model')
            plt.ylabel('Time (milliseconds)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'computational_efficiency.png'))
            
            # Also create a boxplot for more detailed distribution
            plt.figure(figsize=(12, 6))
            
            # Convert data for boxplot
            boxplot_data = []
            labels = []
            colors = []
            
            for model_name, data in thinking_times.items():
                if data['times']:
                    boxplot_data.append(np.array(data['times']) * 1000)  # Convert to ms
                    labels.append(model_name)
                    colors.append('blue' if data['type'] == 'transformer' else 'green')
            
            box = plt.boxplot(boxplot_data, labels=labels, patch_artist=True)
            
            # Color boxes based on model type
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            plt.title('Distribution of Thinking Times by Model')
            plt.ylabel('Time (milliseconds)')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'thinking_time_distribution.png'))
        
        return efficiency_stats
    
    def win_rate_analysis(self):
        """
        Analyze win rates for each model.
        """
        print("Analyzing win rates...")
        
        # Count wins, losses, and draws for each model
        stats = {}
        for name, (model, model_type) in self.models.items():
            stats[name] = {'wins': 0, 'losses': 0, 'draws': 0, 'total': 0, 'type': model_type}
        
        # Process all games
        for game in self.game_data:
            first_player = game['first_player']
            second_player = game['second_player']
            winner = game['winner']
            
            # Update first player stats
            if first_player in stats:
                stats[first_player]['total'] += 1
                if winner == first_player:
                    stats[first_player]['wins'] += 1
                elif winner == 'draw':
                    stats[first_player]['draws'] += 1
                else:
                    stats[first_player]['losses'] += 1
            
            # Update second player stats
            if second_player in stats:
                stats[second_player]['total'] += 1
                if winner == second_player:
                    stats[second_player]['wins'] += 1
                elif winner == 'draw':
                    stats[second_player]['draws'] += 1
                else:
                    stats[second_player]['losses'] += 1
        
        # Calculate win rates
        win_rates = {}
        for model_name, model_stats in stats.items():
            if model_stats['total'] > 0:
                win_rates[model_name] = {
                    'win_rate': model_stats['wins'] / model_stats['total'],
                    'loss_rate': model_stats['losses'] / model_stats['total'],
                    'draw_rate': model_stats['draws'] / model_stats['total'],
                    'games_played': model_stats['total'],
                    'model_type': model_stats['type']
                }
        
        # Save results
        with open(os.path.join(self.output_dir, 'win_rates.json'), 'w') as f:
            json.dump(win_rates, f, indent=2)
        
        self.results['win_rates'] = win_rates
        
        # Visualization
        plt.figure(figsize=(12, 6))
        
        # Prepare data for visualization
        models = []
        win_values = []
        loss_values = []
        draw_values = []
        model_types = []
        
        # Sort models by win rate for better visualization
        sorted_models = sorted(win_rates.items(), key=lambda x: x[1]['win_rate'], reverse=True)
        
        for model_name, stats in sorted_models:
            models.append(model_name)
            win_values.append(stats['win_rate'])
            loss_values.append(stats['loss_rate'])
            draw_values.append(stats['draw_rate'])
            model_types.append(stats['model_type'])
        
        x = np.arange(len(models))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        win_bars = ax.bar(x - width, win_values, width, label='Wins', color='green')
        loss_bars = ax.bar(x, loss_values, width, label='Losses', color='red')
        draw_bars = ax.bar(x + width, draw_values, width, label='Draws', color='gray')
        
        # Add model type indicators
        for i, model_type in enumerate(model_types):
            color = 'blue' if model_type == 'transformer' else 'orange'
            plt.plot(i, -0.05, 'o', markersize=10, color=color)
        
        ax.set_ylabel('Rate')
        ax.set_title('Performance by Model')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        # Add a legend for model types
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Transformer'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Q-Learning')
        ]
        ax.legend(handles=[win_bars, loss_bars, draw_bars] + custom_lines)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'win_rates.png'))
        
        # Also create a head-to-head comparison matrix if we have enough models
        if len(models) > 2:
            self._create_matchup_matrix()
        
        return win_rates
    
    def _create_matchup_matrix(self):
        """Create a matrix showing head-to-head results between models."""
        # Initialize matchup dictionary
        model_names = list(self.models.keys())
        matchups = {name1: {name2: {'wins': 0, 'losses': 0, 'draws': 0, 'total': 0} 
                           for name2 in model_names if name1 != name2} 
                   for name1 in model_names}
        
        # Gather matchup data from games
        for game in self.game_data:
            first_player = game['first_player']
            second_player = game['second_player']
            winner = game['winner']
            
            if first_player in model_names and second_player in model_names:
                # Update matchup stats for first player
                matchups[first_player][second_player]['total'] += 1
                
                if winner == first_player:
                    matchups[first_player][second_player]['wins'] += 1
                elif winner == second_player:
                    matchups[first_player][second_player]['losses'] += 1
                else:
                    matchups[first_player][second_player]['draws'] += 1
                    
                # Update matchup stats for second player
                matchups[second_player][first_player]['total'] += 1
                
                if winner == second_player:
                    matchups[second_player][first_player]['wins'] += 1
                elif winner == first_player:
                    matchups[second_player][first_player]['losses'] += 1
                else:
                    matchups[second_player][first_player]['draws'] += 1
        
        # Calculate win rates for each matchup
        win_rates = np.zeros((len(model_names), len(model_names)))
        
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names):
                if i != j:
                    total_games = matchups[name1][name2]['total']
                    if total_games > 0:
                        win_rates[i, j] = matchups[name1][name2]['wins'] / total_games
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(win_rates, annot=True, cmap="YlGnBu", vmin=0, vmax=1,
                        xticklabels=model_names, yticklabels=model_names)
        plt.title('Win Rate Matrix (Row vs. Column)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'matchup_matrix.png'))
        
        # Save matchup data
        self.results['matchups'] = {name1: {name2: data for name2, data in matches.items()} 
                                  for name1, matches in matchups.items()}
        
        with open(os.path.join(self.output_dir, 'matchups.json'), 'w') as f:
            json.dump(self.results['matchups'], f, indent=2)

    def harmony_analysis(self):
        """
        Analyze harmony patterns across games.
        """
        print("Analyzing harmony patterns...")

        # Extract harmony counts from game data
        harmony_data = []

        for game_idx, game in enumerate(self.game_data):
            winner = game['winner']
            first_player = game['first_player']
            second_player = game['second_player']
            first_type = game.get('first_player_type', 'transformer')
            second_type = game.get('second_player_type', 'transformer')

            # Use harmony progression if available
            harmony_progression = game.get('harmony_progression', [])

            if harmony_progression:
                for move_data in harmony_progression:
                    move_idx = move_data['move']
                    # Determine game stage
                    if move_idx < 10:
                        stage = 'early'
                    elif move_idx < 30:
                        stage = 'mid'
                    else:
                        stage = 'late'

                    harmony_data.append({
                        'Game': game_idx,
                        'Move': move_idx,
                        'Stage': stage,
                        'Player1_Harmony': move_data['player1_harmonies'],
                        'Player2_Harmony': move_data['player2_harmonies'],
                        'Total_Harmony': move_data['total_harmonies'],
                        'Winner': winner,
                        'Player1': first_player,
                        'Player2': second_player,
                        'Player1_Type': first_type,
                        'Player2_Type': second_type
                    })
            else:
                # Extract harmony information from board states if no progression data
                for move_idx, state in enumerate(game.get('board_states', [])):
                    # Determine game stage
                    if move_idx < 10:
                        stage = 'early'
                    elif move_idx < 30:
                        stage = 'mid'
                    else:
                        stage = 'late'

                    # Create a temporary state to analyze harmonies
                    temp_state = SkudPaiShoState()

                    # Convert board back to numpy array if it's a list
                    if isinstance(state, list):
                        board_array = np.array(state, dtype=np.int8)
                        temp_state.board = board_array

                    # Update harmonies in the temporary state
                    if hasattr(temp_state, 'update_harmonies'):
                        temp_state.update_harmonies()

                    # Get harmony counts
                    p1_harmony = 0
                    p2_harmony = 0
                    if hasattr(temp_state, 'harmonies'):
                        p1_harmony = len(temp_state.harmonies.get(1, []))
                        p2_harmony = len(temp_state.harmonies.get(2, []))

                    harmony_data.append({
                        'Game': game_idx,
                        'Move': move_idx,
                        'Stage': stage,
                        'Player1_Harmony': p1_harmony,
                        'Player2_Harmony': p2_harmony,
                        'Total_Harmony': p1_harmony + p2_harmony,
                        'Winner': winner,
                        'Player1': first_player,
                        'Player2': second_player,
                        'Player1_Type': first_type,
                        'Player2_Type': second_type
                    })

        # Convert to DataFrame
        harmony_df = pd.DataFrame(harmony_data)

        # Skip if no data
        if harmony_df.empty:
            print("No harmony data available for analysis.")
            return None

        # Average harmony by game stage
        stage_harmony = harmony_df.groupby('Stage')[
            ['Player1_Harmony', 'Player2_Harmony', 'Total_Harmony']].mean().reset_index()

        # Harmony by winner
        if 'Winner' in harmony_df.columns:
            winner_harmony = harmony_df.groupby('Winner')['Total_Harmony'].mean().reset_index()
        else:
            winner_harmony = pd.DataFrame({'Winner': [], 'Total_Harmony': []})

        # Harmony by model type
        model_type_harmony = pd.DataFrame()
        if 'Player1_Type' in harmony_df.columns and 'Player2_Type' in harmony_df.columns:
            # Get all unique models used in the data
            models_used = set(harmony_df['Player1'].unique()) | set(harmony_df['Player2'].unique())

            # Calculate average harmony generation by model type
            model_harmonies = []
            for model_name in models_used:
                # Get games where this model was player 1
                p1_games = harmony_df[harmony_df['Player1'] == model_name]
                p1_harmony = 0
                if not p1_games.empty:
                    p1_harmony = p1_games['Player1_Harmony'].mean()

                # Get games where this model was player 2
                p2_games = harmony_df[harmony_df['Player2'] == model_name]
                p2_harmony = 0
                if not p2_games.empty:
                    p2_harmony = p2_games['Player2_Harmony'].mean()

                # Calculate weighted average
                total_games = len(p1_games) + len(p2_games)
                if total_games > 0:
                    avg_harmony = (len(p1_games) * p1_harmony + len(p2_games) * p2_harmony) / total_games

                    # Get model type
                    model_type = 'unknown'
                    if model_name in self.models:
                        model_type = self.models[model_name][1]  # Get type from the models dictionary

                    model_harmonies.append({
                        'Model': model_name,
                        'Average_Harmony': avg_harmony,
                        'Type': model_type
                    })

            # Create DataFrame
            if model_harmonies:
                model_type_harmony = pd.DataFrame(model_harmonies)

                # Group by model type
                type_harmony = model_type_harmony.groupby('Type')['Average_Harmony'].mean().reset_index()
                type_harmony.rename(columns={'Type': 'Model_Type'}, inplace=True)

        # Save results
        harmony_results = {
            'stage_harmony': stage_harmony.to_dict('records'),
            'winner_harmony': winner_harmony.to_dict('records')
        }

        if not model_type_harmony.empty:
            harmony_results['model_harmony'] = model_type_harmony.to_dict('records')

        with open(os.path.join(self.output_dir, 'harmony_analysis.json'), 'w') as f:
            json.dump(harmony_results, f, indent=2)

        self.results['harmony_analysis'] = harmony_results

        # Visualization
        plt.figure(figsize=(12, 6))

        if not stage_harmony.empty:
            # Reshape data for grouped bar chart
            stage_harmony_melted = pd.melt(stage_harmony, id_vars=['Stage'],
                                           value_vars=['Player1_Harmony', 'Player2_Harmony', 'Total_Harmony'],
                                           var_name='Harmony_Type', value_name='Harmony')

            # Plot stage harmony
            sns.barplot(x='Stage', y='Harmony', hue='Harmony_Type', data=stage_harmony_melted)
            plt.title('Average Harmony Count by Game Stage')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'harmony_by_stage.png'))

        # Plot harmony by winner
        if not winner_harmony.empty and len(winner_harmony) > 1:
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Winner', y='Total_Harmony', data=winner_harmony)
            plt.title('Average Harmony Count by Winner')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'harmony_by_winner.png'))

        # Plot harmony by model type
        if not model_type_harmony.empty:
            plt.figure(figsize=(12, 6))

            # Sort by harmony score
            model_type_harmony_sorted = model_type_harmony.sort_values('Average_Harmony', ascending=False)

            # Use different colors for different model types
            ax = sns.barplot(x='Model', y='Average_Harmony', data=model_type_harmony_sorted,
                             hue='Type', palette={'transformer': 'blue', 'q_learning': 'green'})
            plt.title('Average Harmony Generation by Model')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'harmony_by_model.png'))

        return harmony_results
    
    def game_length_analysis(self):
        """
        Analyze game length statistics.
        """
        print("Analyzing game length statistics...")
        
        # Extract game lengths and outcomes
        game_lengths = []
        
        for game in self.game_data:
            if 'moves' in game:
                length = len(game['moves'])
                winner = game['winner']
                
                # Get model types if available
                first_type = game.get('first_player_type', 'unknown')
                second_type = game.get('second_player_type', 'unknown')
                
                game_length_data = {
                    'Length': length,
                    'Winner': winner,
                    'First_Player': game['first_player'],
                    'Second_Player': game['second_player']
                }
                
                # Add model types if available
                if 'first_player_type' in game:
                    game_length_data['First_Player_Type'] = first_type
                if 'second_player_type' in game:
                    game_length_data['Second_Player_Type'] = second_type
                
                game_lengths.append(game_length_data)
        
        # Convert to DataFrame
        length_df = pd.DataFrame(game_lengths)
        
        # Skip if no data
        if length_df.empty:
            print("No game length data available.")
            return None
        
        # Calculate statistics
        avg_length = length_df['Length'].mean()
        median_length = length_df['Length'].median()
        
        # Average length by winner
        length_by_winner = length_df.groupby('Winner')['Length'].mean().reset_index()
        
        # Average length by player type (if available)
        length_by_type = None
        if 'First_Player_Type' in length_df.columns:
            # Create a new column indicating if transformer won against q_learning or vice versa
            matchup_results = []
            
            for _, row in length_df.iterrows():
                if row['Winner'] == 'draw':
                    matchup = "Draw"
                elif row['Winner'] == row['First_Player']:
                    if row['First_Player_Type'] == 'transformer' and row['Second_Player_Type'] == 'q_learning':
                        matchup = "Transformer > Q-Learning"
                    elif row['First_Player_Type'] == 'q_learning' and row['Second_Player_Type'] == 'transformer':
                        matchup = "Q-Learning > Transformer"
                    else:
                        matchup = f"{row['First_Player_Type']} > {row['Second_Player_Type']}"
                else:
                    if row['Second_Player_Type'] == 'transformer' and row['First_Player_Type'] == 'q_learning':
                        matchup = "Transformer > Q-Learning"
                    elif row['Second_Player_Type'] == 'q_learning' and row['First_Player_Type'] == 'transformer':
                        matchup = "Q-Learning > Transformer"
                    else:
                        matchup = f"{row['Second_Player_Type']} > {row['First_Player_Type']}"
                
                matchup_results.append({
                    'Matchup': matchup,
                    'Length': row['Length']
                })
            
            # Convert to DataFrame
            matchup_df = pd.DataFrame(matchup_results)
            
            # Calculate average length by matchup
            length_by_type = matchup_df.groupby('Matchup')['Length'].mean().reset_index()
        
        # Calculate average length for first player wins vs second player wins
        first_player_lengths = []
        second_player_lengths = []
        
        for row in game_lengths:
            if row['Winner'] == row['First_Player']:
                first_player_lengths.append(row['Length'])
            elif row['Winner'] == row['Second_Player']:
                second_player_lengths.append(row['Length'])
        
        first_player_avg = np.mean(first_player_lengths) if first_player_lengths else 0
        second_player_avg = np.mean(second_player_lengths) if second_player_lengths else 0
        
        # Save results
        length_results = {
            'average_length': avg_length,
            'median_length': median_length,
            'by_winner': length_by_winner.to_dict('records'),
            'first_player_wins_avg_length': first_player_avg,
            'second_player_wins_avg_length': second_player_avg
        }
        
        if length_by_type is not None:
            length_results['by_matchup_type'] = length_by_type.to_dict('records')
        
        with open(os.path.join(self.output_dir, 'game_length_analysis.json'), 'w') as f:
            json.dump(length_results, f, indent=2)
        
        self.results['game_length_analysis'] = length_results
        
        # Visualization
        plt.figure(figsize=(12, 6))
        
        # Histogram of game lengths
        sns.histplot(length_df['Length'], bins=20)
        plt.axvline(avg_length, color='red', linestyle='--', label=f'Mean: {avg_length:.1f}')
        plt.axvline(median_length, color='green', linestyle='--', label=f'Median: {median_length:.1f}')
        plt.title('Distribution of Game Lengths')
        plt.xlabel('Number of Moves')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'game_length_distribution.png'))
        
        # Game length by winner
        if not length_by_winner.empty:
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Winner', y='Length', data=length_by_winner)
            plt.title('Average Game Length by Winner')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'game_length_by_winner.png'))
        
        # Game length by matchup type
        if length_by_type is not None:
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='Matchup', y='Length', data=length_by_type)
            plt.title('Average Game Length by Matchup Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'game_length_by_matchup.png'))
        
        return length_results
    
    def move_sequence_analysis(self, sequence_length=3):
        """
        Analyze move sequences to identify patterns leading to wins vs. losses.
        
        Args:
            sequence_length: Length of move sequences to analyze
        """
        print("Performing move sequence analysis...")
        
        # Group sequences by model type
        transformer_winning_seqs = defaultdict(int)
        transformer_losing_seqs = defaultdict(int)
        qlearning_winning_seqs = defaultdict(int)
        qlearning_losing_seqs = defaultdict(int)
        
        for game in self.game_data:
            winner = game['winner']
            if winner == 'draw':
                continue
            
            # Get model types
            player_types = {}
            for move_data in game['moves']:
                player = move_data['player']
                if 'player_type' in move_data:
                    player_types[player] = move_data['player_type']
                elif player == game.get('first_player') and 'first_player_type' in game:
                    player_types[player] = game['first_player_type']
                elif player == game.get('second_player') and 'second_player_type' in game:
                    player_types[player] = game['second_player_type']
                else:
                    # Default to transformer if not specified
                    player_types[player] = 'transformer'
            
            # Extract sequences of moves
            for i in range(len(game['moves']) - sequence_length + 1):
                # Create a sequence representation
                sequence = tuple(move['move'] for move in game['moves'][i:i+sequence_length])
                
                # Check if the last move in the sequence was by the winner
                last_player = game['moves'][i+sequence_length-1]['player']
                last_player_type = player_types.get(last_player, 'transformer')
                
                if last_player == winner:
                    # Winning sequence
                    if last_player_type == 'transformer':
                        transformer_winning_seqs[sequence] += 1
                    else:
                        qlearning_winning_seqs[sequence] += 1
                else:
                    # Losing sequence
                    if last_player_type == 'transformer':
                        transformer_losing_seqs[sequence] += 1
                    else:
                        qlearning_losing_seqs[sequence] += 1
        
        # Identify distinctive sequences for each model type
        transformer_distinctive_winning = {}
        transformer_distinctive_losing = {}
        qlearning_distinctive_winning = {}
        qlearning_distinctive_losing = {}
        
        # Process transformer sequences
        for seq, count in transformer_winning_seqs.items():
            losing_count = transformer_losing_seqs.get(seq, 0)
            if count > losing_count * 1.5 and count >= 3:  # Significant win ratio and enough occurrences
                transformer_distinctive_winning[seq] = count
        
        for seq, count in transformer_losing_seqs.items():
            winning_count = transformer_winning_seqs.get(seq, 0)
            if count > winning_count * 1.5 and count >= 3:
                transformer_distinctive_losing[seq] = count
        
        # Process Q-learning sequences
        for seq, count in qlearning_winning_seqs.items():
            losing_count = qlearning_losing_seqs.get(seq, 0)
            if count > losing_count * 1.5 and count >= 3:
                qlearning_distinctive_winning[seq] = count
        
        for seq, count in qlearning_losing_seqs.items():
            winning_count = qlearning_winning_seqs.get(seq, 0)
            if count > winning_count * 1.5 and count >= 3:
                qlearning_distinctive_losing[seq] = count
        
        # Save results
        sequence_results = {
            'transformer': {
                'winning_sequences': {str(k): v for k, v in sorted(transformer_distinctive_winning.items(), 
                                                key=lambda x: x[1], reverse=True)[:15]},
                'losing_sequences': {str(k): v for k, v in sorted(transformer_distinctive_losing.items(), 
                                               key=lambda x: x[1], reverse=True)[:15]}
            },
            'q_learning': {
                'winning_sequences': {str(k): v for k, v in sorted(qlearning_distinctive_winning.items(), 
                                                key=lambda x: x[1], reverse=True)[:15]},
                'losing_sequences': {str(k): v for k, v in sorted(qlearning_distinctive_losing.items(), 
                                               key=lambda x: x[1], reverse=True)[:15]}
            }
        }
        
        with open(os.path.join(self.output_dir, 'move_sequences.json'), 'w') as f:
            json.dump(sequence_results, f, indent=2)
        
        self.results['move_sequences'] = sequence_results
        
        # Visualization
        # Create DataFrame for visualization - Transformer
        transformer_data = []
        
        for seq, count in sorted(transformer_distinctive_winning.items(), key=lambda x: x[1], reverse=True)[:10]:
            transformer_data.append({
                'Sequence': " → ".join(map(str, seq)), 
                'Count': count, 
                'Type': 'Winning', 
                'Model': 'Transformer'
            })
        
        for seq, count in sorted(transformer_distinctive_losing.items(), key=lambda x: x[1], reverse=True)[:10]:
            transformer_data.append({
                'Sequence': " → ".join(map(str, seq)), 
                'Count': count, 
                'Type': 'Losing', 
                'Model': 'Transformer'
            })
        
        # Create DataFrame for visualization - Q-Learning
        qlearning_data = []
        
        for seq, count in sorted(qlearning_distinctive_winning.items(), key=lambda x: x[1], reverse=True)[:10]:
            qlearning_data.append({
                'Sequence': " → ".join(map(str, seq)), 
                'Count': count, 
                'Type': 'Winning', 
                'Model': 'Q-Learning'
            })
        
        for seq, count in sorted(qlearning_distinctive_losing.items(), key=lambda x: x[1], reverse=True)[:10]:
            qlearning_data.append({
                'Sequence': " → ".join(map(str, seq)), 
                'Count': count, 
                'Type': 'Losing', 
                'Model': 'Q-Learning'
            })
        
        # Combine dataframes
        combined_df = pd.DataFrame(transformer_data + qlearning_data)
        
        if not combined_df.empty:
            # Plot transformer and Q-learning on separate subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
            
            # Filter data
            transformer_df = combined_df[combined_df['Model'] == 'Transformer']
            qlearning_df = combined_df[combined_df['Model'] == 'Q-Learning']
            
            # Plot transformer data
            if not transformer_df.empty:
                sns.barplot(x='Count', y='Sequence', hue='Type', data=transformer_df, 
                          ax=ax1, palette={'Winning': 'green', 'Losing': 'red'})
                ax1.set_title(f'Top Transformer {sequence_length}-Move Sequences')
                ax1.set_xlabel('Occurrence Count')
            
            # Plot Q-learning data
            if not qlearning_df.empty:
                sns.barplot(x='Count', y='Sequence', hue='Type', data=qlearning_df, 
                          ax=ax2, palette={'Winning': 'green', 'Losing': 'red'})
                ax2.set_title(f'Top Q-Learning {sequence_length}-Move Sequences')
                ax2.set_xlabel('Occurrence Count')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'move_sequences.png'))
        
        return sequence_results
    
    def run_functional_evaluation(self):
        """Run all functional evaluation analyses."""
        print("Running enhanced evaluation suite...")
        
        # Run functional analyses
        win_rate_results = self.win_rate_analysis()
        efficiency_results = self.computational_efficiency_analysis()
        harmony_results = self.harmony_analysis()
        length_results = self.game_length_analysis()
        
        # Run additional analyses
        try:
            move_seq_results = self.move_sequence_analysis()
        except Exception as e:
            print(f"Move sequence analysis failed: {e}")
        
        # Create summary report
        self._create_summary_report()
        
        return self.results
    
    def _create_summary_report(self):
        """Create a summary report of all analyses."""
        # Extract model types
        model_types = {name: model_type for name, (model, model_type) in self.models.items()}
        
        report = {
            'summary': {
                'models_evaluated': list(self.models.keys()),
                'model_types': model_types,
                'num_games_analyzed': len(self.game_data),
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'game_type': 'Skud Pai Sho'
            }
        }
        
        # Add analysis results
        report.update(self.results)
        
        # Save report
        with open(os.path.join(self.output_dir, 'evaluation_summary.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create HTML report
        self._create_html_report(model_types)
        
        return report

    def _create_html_report(self, model_types):
        """Create an HTML report with embedded visualizations."""
        # Create a color-coded model list based on model types
        model_list = []
        for name, type_info in model_types.items():
            color = 'blue' if type_info == 'transformer' else 'green'
            model_list.append(f'<span style="color:{color}">{name} ({type_info})</span>')

        models_html = ", ".join(model_list)

        html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Enhanced Skud Pai Sho AI Evaluation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333; }
                    h2 { color: #555; margin-top: 30px; }
                    .section { margin-bottom: 40px; }
                    .image-container { display: flex; flex-wrap: wrap; gap: 20px; }
                    .image-container img { max-width: 100%; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    tr:nth-child(even) { background-color: #f9f9f9; }
                    .legend { display: flex; gap: 15px; margin-bottom: 20px; }
                    .legend-item { display: flex; align-items: center; }
                    .legend-color { width: 15px; height: 15px; margin-right: 5px; }
                </style>
            </head>
            <body>
                <h1>Enhanced Skud Pai Sho AI Evaluation Report</h1>
                <p>Generated on: {timestamp}</p>

                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: blue;"></div>
                        <span>Transformer Model</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: green;"></div>
                        <span>Q-Learning Model</span>
                    </div>
                </div>

                <p>Models evaluated: {models}</p>
                <p>Number of games analyzed: {num_games}</p>

                <div class="section">
                    <h2>Win Rate Analysis</h2>
                    <p>Performance comparison between models:</p>
                    <div class="image-container">
                        <img src="win_rates.png" alt="Win Rate Analysis">
                        <img src="matchup_matrix.png" alt="Matchup Matrix" onerror="this.style.display='none'">
                    </div>
                </div>

                <div class="section">
                    <h2>Game Length Analysis</h2>
                    <p>Analysis of game duration patterns:</p>
                    <div class="image-container">
                        <img src="game_length_distribution.png" alt="Game Length Distribution">
                        <img src="game_length_by_winner.png" alt="Game Length by Winner">
                        <img src="game_length_by_matchup.png" alt="Game Length by Matchup" onerror="this.style.display='none'">
                    </div>
                </div>

                <div class="section">
                    <h2>Move Sequence Analysis</h2>
                    <p>Patterns of moves that lead to wins vs. losses:</p>
                    <div class="image-container">
                        <img src="move_sequences.png" alt="Move Sequences Analysis" onerror="this.style.display='none'">
                    </div>
                </div>

                <div class="section">
                    <h2>Harmony Analysis</h2>
                    <p>Analysis of harmony patterns and their relation to victory:</p>
                    <div class="image-container">
                        <img src="harmony_by_stage.png" alt="Harmony by Game Stage">
                        <img src="harmony_by_winner.png" alt="Harmony by Winner" onerror="this.style.display='none'">
                        <img src="harmony_by_model.png" alt="Harmony by Model" onerror="this.style.display='none'">
                    </div>
                </div>

                <div class="section">
                    <h2>Computational Efficiency</h2>
                    <p>Comparison of thinking time between models:</p>
                    <div class="image-container">
                        <img src="computational_efficiency.png" alt="Computational Efficiency">
                        <img src="thinking_time_distribution.png" alt="Thinking Time Distribution">
                    </div>
                </div>

                <div class="section">
                    <h2>Conclusion</h2>
                    <p>This evaluation compared the performance of transformer-based neural network models and Q-learning 
                    agents in playing Skud Pai Sho. The analysis provides insights into the strengths and weaknesses 
                    of each approach, as well as their computational efficiency and strategic patterns.</p>
                </div>
            </body>
            </html>
            """.format(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            models=models_html,
            num_games=len(self.game_data)
        )

        with open(os.path.join(self.output_dir, 'evaluation_report.html'), 'w') as f:
            f.write(html)