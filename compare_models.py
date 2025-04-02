#!/usr/bin/env python
# compare_models.py - Script to analyze Transformer vs Q-Learning model performance
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import pandas as pd
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm
import pickle

from game.state import SkudPaiShoState
from ai.model import SkudPaiShoTransformer
# Import Q-learning implementation
from q_learning.q_learning_ai import SkudPaiShoQLearning


def load_q_learning_model(model_path):
    """Load a Q-learning model from a pickle file."""
    print(f"Loading Q-learning model from {model_path}")
    q_model = SkudPaiShoQLearning()
    q_model.load(model_path)
    return q_model


# Add these adapter methods to provide the interface expected by the comparison script
def get_move_q_learning(q_model, state, valid_moves=None):
    """
    Adapter function to interface with Q-learning model.
    Returns a move and the estimated value.
    """
    if valid_moves is None:
        valid_moves = state.get_valid_moves()

    if not valid_moves:
        return None, 0.0

    # Get state key
    state_key = q_model.get_state_key(state)

    # Get Q-values for all valid actions
    q_values = {move: q_model.q_table[state_key][q_model.get_action_key(move)] for move in valid_moves}

    # Find best move
    best_move = max(q_values.items(), key=lambda x: x[1])[0]
    best_value = q_values[best_move]

    # Scale value to be in range [-1, 1] similar to transformer values
    # This is approximate since Q-values have different scales
    normalized_value = max(min(best_value / 10.0, 1.0), -1.0)

    return best_move, normalized_value


def evaluate_move_q_learning(q_model, state, move):
    """
    Evaluate a specific move with the Q-learning model.
    Returns the Q-value for that state-action pair.
    """
    state_key = q_model.get_state_key(state)
    action_key = q_model.get_action_key(move)
    q_value = q_model.q_table[state_key][action_key]

    # Normalize to [-1, 1] range
    normalized_value = max(min(q_value / 10.0, 1.0), -1.0)

    return normalized_value


def play_game(transformer_model, q_model, transformer_mcts=0, temperature=0.0, deterministic=True):
    """
    Play a game between transformer and Q-learning models.

    Args:
        transformer_model: The transformer model
        q_model: The Q-learning model
        transformer_mcts: Number of MCTS simulations for transformer (0 for direct policy)
        temperature: Temperature parameter for move selection
        deterministic: Whether to use deterministic move selection

    Returns:
        Dictionary with game statistics and move history
    """
    from ai.utils import get_ai_move

    state = SkudPaiShoState()
    move_history = []
    game_metrics = {
        "transformer_eval": [],
        "q_model_eval": [],
        "move_confidence": [],
        "similar_move_choices": 0,
        "different_move_choices": 0,
        "transformer_move_times": [],
        "q_model_move_times": [],
        "harmonies_by_turn": [],
        "board_control": []
    }

    while not state.is_game_over() and state.turn_number < 200:  # Max 200 moves to prevent infinite games
        current_player = state.current_player

        # Get valid moves
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            break

        if current_player == 1:  # Transformer's turn
            start_time = time.time()

            # Get transformer move
            move, value = get_ai_move(
                transformer_model,
                state,
                temperature=temperature,
                deterministic=deterministic,
                mcts_simulations=transformer_mcts
            )

            move_time = time.time() - start_time
            game_metrics["transformer_move_times"].append(move_time)
            game_metrics["transformer_eval"].append(value)

            # For comparison, also get Q-model's evaluation of this position
            q_start_time = time.time()
            q_move, q_value = get_move_q_learning(q_model, state, valid_moves)
            q_move_time = time.time() - q_start_time

            game_metrics["q_model_eval"].append(q_value)

            # Check if models would make the same move
            if move == q_move:
                game_metrics["similar_move_choices"] += 1
            else:
                game_metrics["different_move_choices"] += 1

            # Record move confidence (how strongly the transformer prefers the chosen move)
            # For simplicity, we'll use 0.8 as a placeholder
            confidence = 0.8  # Placeholder
            game_metrics["move_confidence"].append(confidence)

        else:  # Q-model's turn
            start_time = time.time()

            # Get Q-model move
            move, q_value = get_move_q_learning(q_model, state, valid_moves)

            move_time = time.time() - start_time
            game_metrics["q_model_move_times"].append(move_time)
            game_metrics["q_model_eval"].append(q_value)

            # For comparison, also get transformer's evaluation
            t_start_time = time.time()
            # Get transformer evaluation without actually making a move
            _, value = get_ai_move(
                transformer_model,
                state,
                temperature=0,
                deterministic=True,
                mcts_simulations=0  # No MCTS needed for just evaluation
            )
            t_move_time = time.time() - t_start_time

            game_metrics["transformer_eval"].append(value)

        # Record move
        move_history.append({
            "turn": state.turn_number,
            "player": current_player,
            "move": str(move),
            "board_state": state.copy()  # For later analysis
        })

        # Make the move
        state.make_move(move)

        # Record board metrics
        game_metrics["harmonies_by_turn"].append(state.count_harmonies())
        game_metrics["board_control"].append(calculate_board_control(state))

    # Game over
    result = state.get_reward(1)  # From transformer's perspective

    game_summary = {
        "winner": 1 if result > 0 else (2 if result < 0 else 0),  # 0 for draw
        "num_moves": state.turn_number,
        "final_score": result,
        "metrics": game_metrics,
        "move_history": move_history
    }

    return game_summary


def calculate_board_control(state):
    """Calculate a simple board control metric for each player."""
    p1_control = 0
    p2_control = 0

    # Adapt to your actual board representation
    for y in range(state.board.shape[0]):
        for x in range(state.board.shape[1]):
            piece = state.board[y, x]
            if piece != 0:  # Not empty
                # Determine piece owner based on piece type
                # Assuming positive values are player 1, negative are player 2
                owner = 1 if piece > 0 else 2

                # Central positions are worth more
                center_distance = abs(x - state.board.shape[1] // 2) + abs(y - state.board.shape[0] // 2)
                value = 10 - center_distance

                if owner == 1:
                    p1_control += value
                else:
                    p2_control += value

    # Return normalized control values
    total = p1_control + p2_control
    if total == 0:
        return 0.5  # Equal control (empty board)
    return p1_control / total


def analyze_position(transformer_model, q_model, state, num_top_moves=3):
    """Analyze a position with both models and compare their move preferences."""
    # Get valid moves
    valid_moves = state.get_valid_moves()
    if not valid_moves:
        return None

    # Get transformer's evaluation
    state_tensor = torch.tensor(state.encode_for_network(), dtype=torch.float32).unsqueeze(0)
    device = next(transformer_model.parameters()).device
    state_tensor = state_tensor.to(device)

    with torch.no_grad():
        policy_logits, value = transformer_model(state_tensor)

    # Convert to probability distribution
    policy = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

    from ai.training import action_to_index

    # Match policy to valid moves
    valid_indices = [action_to_index(move) for move in valid_moves]

    # Extract probabilities for valid moves
    t_move_probs = []
    for i, move in enumerate(valid_moves):
        idx = valid_indices[i]
        if idx < len(policy):
            t_move_probs.append((move, policy[idx], i))

    # Sort by probability (highest first)
    t_move_probs.sort(key=lambda x: x[1], reverse=True)

    # Get Q-model's evaluation for the same moves
    q_move_probs = []
    for move in valid_moves:
        # Evaluate each move with Q-model
        q_value = evaluate_move_q_learning(q_model, state, move)
        q_move_probs.append((move, q_value))

    # Sort by value (highest first)
    q_move_probs.sort(key=lambda x: x[1], reverse=True)

    # Filter to top moves
    t_top_moves = t_move_probs[:num_top_moves] if t_move_probs else []
    q_top_moves = q_move_probs[:num_top_moves] if q_move_probs else []

    # Build comparison data
    move_comparison = {
        "transformer_value": value.item(),
        "transformer_top_moves": t_top_moves,
        "q_model_top_moves": q_top_moves,
        "agreement": t_top_moves[0][0] == q_top_moves[0][0] if t_top_moves and q_top_moves else False,
        "rank_correlation": calculate_rank_correlation(t_move_probs, q_move_probs)
    }

    return move_comparison


def calculate_rank_correlation(t_move_probs, q_move_probs):
    """Calculate Spearman rank correlation between the two models' move rankings."""
    # Create dictionaries mapping moves to their ranks
    t_ranks = {str(move): i for i, (move, _, _) in enumerate(t_move_probs)}
    q_ranks = {str(move): i for i, (move, _) in enumerate(q_move_probs)}

    # Get common moves
    common_moves = set(t_ranks.keys()).intersection(set(q_ranks.keys()))

    if len(common_moves) < 2:
        return 0  # Not enough moves for meaningful correlation

    # Extract ranks for common moves
    t_common_ranks = [t_ranks[move] for move in common_moves]
    q_common_ranks = [q_ranks[move] for move in common_moves]

    # Calculate Spearman correlation
    from scipy.stats import spearmanr
    corr, _ = spearmanr(t_common_ranks, q_common_ranks)

    return corr if not np.isnan(corr) else 0


def analyze_move_patterns(game_summaries):
    """Analyze common patterns in how the models play."""
    transformer_moves = defaultdict(int)
    q_model_moves = defaultdict(int)

    # Categorize moves by type and frequency
    for game in game_summaries:
        for move_info in game["move_history"]:
            move_str = move_info["move"]
            player = move_info["player"]

            # Simplify move representation for pattern detection
            # Example: "plant fire 3 4" -> "plant_fire"
            simplified = simplify_move(move_str)

            if player == 1:  # Transformer
                transformer_moves[simplified] += 1
            else:  # Q-model
                q_model_moves[simplified] += 1

    # Get most common moves for each model
    transformer_common = sorted(transformer_moves.items(), key=lambda x: x[1], reverse=True)[:10]
    q_model_common = sorted(q_model_moves.items(), key=lambda x: x[1], reverse=True)[:10]

    # Calculate unique moves (only used by one model)
    transformer_unique = set(transformer_moves.keys()) - set(q_model_moves.keys())
    q_model_unique = set(q_model_moves.keys()) - set(transformer_moves.keys())

    patterns = {
        "transformer_common_moves": transformer_common,
        "q_model_common_moves": q_model_common,
        "transformer_unique_moves": list(transformer_unique),
        "q_model_unique_moves": list(q_model_unique),
        "move_diversity": {
            "transformer": len(transformer_moves),
            "q_model": len(q_model_moves)
        }
    }

    return patterns


def simplify_move(move_str):
    """Simplify a move string for pattern analysis."""
    # Extract the move type and piece type if planting
    parts = move_str.strip().replace("(", "").replace(")", "").replace("'", "").split(",")

    if parts[0].strip() == "plant":
        return f"plant_{parts[1].strip()}"
    elif parts[0].strip() == "move":
        # For moves, calculate if it's short/medium/long distance
        try:
            from_x, from_y = int(parts[1].strip()), int(parts[2].strip())
            to_x, to_y = int(parts[3].strip()), int(parts[4].strip())

            # Calculate Manhattan distance
            distance = abs(to_x - from_x) + abs(to_y - from_y)

            if distance <= 2:
                category = "short"
            elif distance <= 4:
                category = "medium"
            else:
                category = "long"

            return f"move_{category}"
        except:
            return "move_other"

    return "other"


def visualize_comparisons(analysis_results, output_dir):
    """Create visualizations of the model comparison results."""
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    game_summaries = analysis_results["game_summaries"]
    pattern_analysis = analysis_results["pattern_analysis"]
    position_analyses = analysis_results["position_analyses"]

    # 1. Win rates chart
    plt.figure(figsize=(8, 6))

    wins_transformer = analysis_results["overall_stats"]["transformer_wins"]
    wins_q = analysis_results["overall_stats"]["q_model_wins"]
    draws = analysis_results["overall_stats"]["draws"]

    plt.bar(['Transformer', 'Q-Learning', 'Draws'], [wins_transformer, wins_q, draws], color=['blue', 'green', 'gray'])
    plt.title('Win Distribution')
    plt.ylabel('Number of Games')
    plt.savefig(os.path.join(output_dir, 'win_distribution.png'))
    plt.close()

    # 2. Game length histogram
    plt.figure(figsize=(10, 6))
    game_lengths = [game["num_moves"] for game in game_summaries]

    transformer_wins = [game["num_moves"] for game in game_summaries if game["winner"] == 1]
    q_wins = [game["num_moves"] for game in game_summaries if game["winner"] == 2]
    draw_lengths = [game["num_moves"] for game in game_summaries if game["winner"] == 0]

    plt.hist([transformer_wins, q_wins, draw_lengths], bins=15,
             label=['Transformer Wins', 'Q-Learning Wins', 'Draws'],
             alpha=0.7, color=['blue', 'green', 'gray'])
    plt.title('Game Length Distribution by Outcome')
    plt.xlabel('Number of Moves')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'game_length_distribution.png'))
    plt.close()

    # 3. Move time comparison
    plt.figure(figsize=(10, 6))

    # Collect move times across all games
    transformer_times = []
    q_times = []

    for game in game_summaries:
        transformer_times.extend(game["metrics"]["transformer_move_times"])
        q_times.extend(game["metrics"]["q_model_move_times"])

    plt.boxplot([transformer_times, q_times], labels=['Transformer', 'Q-Learning'])
    plt.title('Move Decision Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'move_time_comparison.png'))
    plt.close()

    # 4. Evaluation correlation
    plt.figure(figsize=(10, 6))

    # Get paired evaluations from both models
    paired_evals = []

    for game in game_summaries:
        t_evals = game["metrics"]["transformer_eval"]
        q_evals = game["metrics"]["q_model_eval"]

        # Only use paired evaluations
        min_len = min(len(t_evals), len(q_evals))
        paired_evals.extend([(t_evals[i], q_evals[i]) for i in range(min_len)])

    t_vals = [x[0] for x in paired_evals]
    q_vals = [x[1] for x in paired_evals]

    plt.scatter(t_vals, q_vals, alpha=0.5)
    plt.xlabel('Transformer Evaluation')
    plt.ylabel('Q-Learning Evaluation')
    plt.title('Position Evaluation Correlation')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'evaluation_correlation.png'))
    plt.close()

    # 5. Move pattern comparison
    plt.figure(figsize=(12, 8))

    # Convert move pattern data to dataframe
    t_common = pattern_analysis["transformer_common_moves"]
    q_common = pattern_analysis["q_model_common_moves"]

    t_df = pd.DataFrame(t_common, columns=['move', 'count'])
    q_df = pd.DataFrame(q_common, columns=['move', 'count'])

    t_df['model'] = 'Transformer'
    q_df['model'] = 'Q-Learning'

    df = pd.concat([t_df, q_df])

    # Create grouped bar chart
    ax = sns.barplot(x='move', y='count', hue='model', data=df)
    plt.title('Most Common Move Patterns by Model')
    plt.ylabel('Frequency')
    plt.xlabel('Move Pattern')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'move_pattern_comparison.png'))
    plt.close()

    # 6. Harmony progression chart
    plt.figure(figsize=(12, 6))

    # Plot harmony progression for a few sample games
    sample_games = game_summaries[:min(5, len(game_summaries))]

    for i, game in enumerate(sample_games):
        harmonies = game["metrics"]["harmonies_by_turn"]
        plt.plot(range(len(harmonies)), harmonies, label=f'Game {i + 1}')

    plt.title('Harmony Progression During Games')
    plt.xlabel('Turn Number')
    plt.ylabel('Harmony Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'harmony_progression.png'))
    plt.close()

    # 7. Board control chart
    plt.figure(figsize=(12, 6))

    # Plot board control progression for a few sample games
    for i, game in enumerate(sample_games):
        control = game["metrics"]["board_control"]
        plt.plot(range(len(control)), control, label=f'Game {i + 1}')

    plt.title('Board Control Progression (Higher = Transformer Advantage)')
    plt.xlabel('Turn Number')
    plt.ylabel('Board Control (0-1)')
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'board_control_progression.png'))
    plt.close()

    # 8. Agreement on top moves
    agreement_rates = []
    rank_correlations = []

    for analysis in position_analyses:
        if analysis:  # Skip None values
            agreement_rates.append(1 if analysis["agreement"] else 0)
            rank_correlations.append(analysis["rank_correlation"])

    plt.figure(figsize=(10, 6))
    plt.hist(rank_correlations, bins=10)
    plt.title('Distribution of Move Ranking Correlation')
    plt.xlabel('Spearman Rank Correlation')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'move_ranking_correlation.png'))
    plt.close()

    # 9. Agreement rate on best move
    plt.figure(figsize=(8, 6))
    agreement_pct = sum(agreement_rates) / len(agreement_rates) if agreement_rates else 0
    plt.bar(['Agreement', 'Disagreement'], [agreement_pct, 1 - agreement_pct])
    plt.title('Best Move Agreement Rate')
    plt.ylabel('Percentage')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'best_move_agreement.png'))
    plt.close()

    # Save a summary of the analysis results
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Transformer vs Q-Learning Analysis Summary\n")
        f.write("========================================\n\n")

        f.write(f"Games Analyzed: {len(game_summaries)}\n")
        f.write(f"Transformer Wins: {wins_transformer} ({wins_transformer / len(game_summaries) * 100:.1f}%)\n")
        f.write(f"Q-Learning Wins: {wins_q} ({wins_q / len(game_summaries) * 100:.1f}%)\n")
        f.write(f"Draws: {draws} ({draws / len(game_summaries) * 100:.1f}%)\n\n")

        f.write(f"Average Game Length: {np.mean(game_lengths):.1f} moves\n\n")

        f.write("Move Time Analysis:\n")
        f.write(f"  Transformer Avg. Move Time: {np.mean(transformer_times):.4f} seconds\n")
        f.write(f"  Q-Learning Avg. Move Time: {np.mean(q_times):.4f} seconds\n")
        f.write(f"  Transformer Speed Advantage: {np.mean(q_times) / np.mean(transformer_times):.1f}x\n\n")

        f.write("Move Pattern Analysis:\n")
        f.write(f"  Transformer Unique Move Types: {pattern_analysis['move_diversity']['transformer']}\n")
        f.write(f"  Q-Learning Unique Move Types: {pattern_analysis['move_diversity']['q_model']}\n")
        f.write(f"  Best Move Agreement Rate: {agreement_pct * 100:.1f}%\n")
        f.write(f"  Average Move Ranking Correlation: {np.mean(rank_correlations):.3f}\n\n")

        f.write("Top 3 Move Patterns for Transformer:\n")
        for move, count in pattern_analysis["transformer_common_moves"][:3]:
            f.write(f"  {move}: {count} times\n")

        f.write("\nTop 3 Move Patterns for Q-Learning:\n")
        for move, count in pattern_analysis["q_model_common_moves"][:3]:
            f.write(f"  {move}: {count} times\n")

    print(f"Analysis visualizations saved to {output_dir}")
    return summary_file

    # Compile analysis results
    analysis_results = {
        "overall_stats": overall_stats,
        "game_summaries": game_summaries,
        "pattern_analysis": pattern_analysis,
        "position_analyses": position_analyses
    }

    # Save raw analysis results
    results_file = os.path.join(args.output_dir, 'analysis_results.json')

    # Convert game summaries to serializable format
    serializable_summaries = []
    for game in game_summaries:
        serializable_game = {
            "winner": game["winner"],
            "num_moves": game["num_moves"],
            "final_score": game["final_score"],
            "metrics": {
                k: v for k, v in game["metrics"].items()
                if k not in ['board_states']
            },
            "move_history": [
                {
                    "turn": move["turn"],
                    "player": move["player"],
                    "move": move["move"]
                }
                for move in game["move_history"]
            ]
        }
        serializable_summaries.append(serializable_game)

    serializable_results = {
        "overall_stats": overall_stats,
        "pattern_analysis": pattern_analysis,
        # Exclude position_analyses as it contains non-serializable objects
    }

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    # Create visualizations
    print("Creating visualizations...")
    summary_file = visualize_comparisons(analysis_results, args.output_dir)

    print(f"Analysis complete! Results saved to {args.output_dir}")
    print(f"See {summary_file} for a summary of the analysis")


def main():
    parser = argparse.ArgumentParser(description='Compare Transformer and Q-Learning models in Skud Pai Sho')

    # Model parameters
    parser.add_argument('--transformer_model', type=str, required=True, help='Path to transformer model file')
    parser.add_argument('--q_model', type=str, required=True, help='Path to Q-learning model file')

    # Comparison parameters
    parser.add_argument('--num_games', type=int, default=50, help='Number of games to play')
    parser.add_argument('--transformer_mcts', type=int, default=0,
                        help='Number of MCTS simulations for transformer (0 for direct policy)')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic move selection')

    # Analysis parameters
    parser.add_argument('--analyze_positions', type=int, default=20,
                        help='Number of positions to analyze in detail')
    parser.add_argument('--num_top_moves', type=int, default=3,
                        help='Number of top moves to analyze per position')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='model_comparison_results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load transformer model
    print(f"Loading transformer model from {args.transformer_model}")
    checkpoint = torch.load(args.transformer_model, map_location=device)

    # Create sample state to determine input channels
    sample_state = SkudPaiShoState()
    encoded_state = sample_state.encode_for_network()
    input_channels = encoded_state.shape[0]

    # Get model architecture from checkpoint or use default
    d_model = checkpoint.get('d_model', 256)
    nhead = checkpoint.get('nhead', 8)
    num_layers = checkpoint.get('num_layers', 3)
    dropout = checkpoint.get('dropout', 0.0)  # Use 0 dropout for evaluation

    transformer_model = SkudPaiShoTransformer(
        input_channels=input_channels,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )
    transformer_model.load_state_dict(checkpoint['model_state_dict'])
    transformer_model.eval()  # Set to evaluation mode
    transformer_model = transformer_model.to(device)

    # Load Q-learning model
    q_model = load_q_learning_model(args.q_model)

    # Set low exploration rate for consistent move choices
    q_model.exploration_rate = q_model.min_exploration_rate

    # Play games and collect data
    print(f"Playing {args.num_games} games for comparison...")
    game_summaries = []

    for i in tqdm(range(args.num_games)):
        game_summary = play_game(
            transformer_model,
            q_model,
            transformer_mcts=args.transformer_mcts,
            temperature=0.0 if args.deterministic else 0.5,
            deterministic=args.deterministic
        )
        game_summaries.append(game_summary)

    # Analyze move patterns
    print("Analyzing move patterns...")
    pattern_analysis = analyze_move_patterns(game_summaries)

    # Analyze specific positions
    print(f"Analyzing {args.analyze_positions} positions in detail...")
    position_analyses = []

    # Select positions from played games
    positions_analyzed = 0
    game_indices = np.random.choice(len(game_summaries), min(len(game_summaries), args.analyze_positions),
                                    replace=False)

    for game_idx in game_indices:
        game = game_summaries[game_idx]
        if game["move_history"]:
            # Select a random move from the game
            move_idx = np.random.randint(0, len(game["move_history"]))
            position = game["move_history"][move_idx]["board_state"]

            # Analyze position
            analysis = analyze_position(
                transformer_model,
                q_model,
                position,
                num_top_moves=args.num_top_moves
            )
            position_analyses.append(analysis)
            positions_analyzed += 1

    print(f"Analyzed {positions_analyzed} positions")

    # Calculate overall statistics
    transformer_wins = sum(1 for game in game_summaries if game["winner"] == 1)
    q_model_wins = sum(1 for game in game_summaries if game["winner"] == 2)
    draws = sum(1 for game in game_summaries if game["winner"] == 0)

    overall_stats = {
        "transformer_wins": transformer_wins,
        "q_model_wins": q_model_wins,
        "draws": draws,
        "transformer_win_rate": transformer_wins / args.num_games,
        "q_model_win_rate": q_model_wins / args.num_games,
        "draw_rate": draws / args.num_games
    }

    # Compile analysis results
    analysis_results = {
        "overall_stats": overall_stats,
        "game_summaries": game_summaries,
        "pattern_analysis": pattern_analysis,
        "position_analyses": position_analyses
    }

    # Save raw analysis results
    results_file = os.path.join(args.output_dir, 'analysis_results.json')

    # Convert game summaries to serializable format
    serializable_summaries = []
    for game in game_summaries:
        serializable_game = {
            "winner": game["winner"],
            "num_moves": game["num_moves"],
            "final_score": game["final_score"],
            "metrics": {
                k: v for k, v in game["metrics"].items()
                if k not in ['board_states']
            },
            "move_history": [
                {
                    "turn": move["turn"],
                    "player": move["player"],
                    "move": move["move"]
                }
                for move in game["move_history"]
            ]
        }
        serializable_summaries.append(serializable_game)

    serializable_results = {
        "overall_stats": overall_stats,
        "pattern_analysis": pattern_analysis,
        "game_summaries": serializable_summaries
    }

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    # Create visualizations
    print("Creating visualizations...")
    summary_file = visualize_comparisons(analysis_results, args.output_dir)

    print(f"Analysis complete! Results saved to {args.output_dir}")
    print(f"See {summary_file} for a summary of the analysis")


if __name__ == "__main__":
    main()