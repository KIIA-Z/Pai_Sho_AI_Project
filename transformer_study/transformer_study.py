# transformer_study.py
import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import defaultdict
# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add it to the path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from game.state import SkudPaiShoState
from game.state import SkudPaiShoState, TileType, BOARD_SIZE
from game.display import display_board, move_to_string
from ai_deprecated.model import SkudPaiShoTransformer
from ai_deprecated.training import self_play, train, action_to_index
from ai_deprecated.utils import get_ai_move


class SkudPaiShoTransformerStudy:
    def __init__(self, study_name="transformer_study"):
        """Initialize the transformer study framework."""
        self.study_name = study_name
        os.makedirs("study_results", exist_ok=True)
        self.results_dir = os.path.join("study_results", study_name)
        os.makedirs(self.results_dir, exist_ok=True)

        # Study parameters
        self.parameters = {}
        self.metrics = {}

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = None

    def initialize_model(self, d_model=256, nhead=8, num_layers=6, dropout=0.1):
        """Initialize the transformer model with specific architecture parameters."""
        # Calculate input channels based on encoding
        input_channels = len(
            TileType)  + 4  # All tile types (except EMPTY) + turn + player + harmonies(2) + board mask

        self.model = SkudPaiShoTransformer(
            input_channels=input_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)

        self.parameters["model"] = {
            "input_channels": input_channels,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout
        }

        return self.model

    def load_model(self, model_path):
        """Load a pre-trained model."""
        if self.model is None:
            self.initialize_model()

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from {model_path}")
        return self.model

    def save_model(self, filename="transformer_model.pth"):
        """Save the current model."""
        if self.model is None:
            print("No model to save.")
            return

        model_path = os.path.join(self.results_dir, filename)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def train_model(self, iterations=50, games_per_iteration=20, epochs_per_iteration=5,
                    batch_size=128, lr=0.001, save_interval=10):
        """Train the transformer model and record performance metrics."""
        if self.model is None:
            self.initialize_model()

        print(f"Training transformer model for {iterations} iterations")
        print(f"Games per iteration: {games_per_iteration}, Epochs per iteration: {epochs_per_iteration}")

        # Record training start time
        start_time = time.time()

        # Initialize metrics tracking
        iteration_numbers = []
        policy_losses = []
        value_losses = []
        game_lengths = []
        harmony_counts = []
        training_times = []

        for iteration in range(iterations):
            iter_start_time = time.time()
            print(f"\nIteration {iteration + 1}/{iterations}")

            # Generate self-play games
            print(f"Generating {games_per_iteration} self-play games...")
            game_records = self_play(self.model, num_games=games_per_iteration)

            # Record game statistics
            avg_game_length = sum(len(record[0].history) for record in game_records) / len(game_records)
            avg_harmonies = sum(
                len(record[0].harmonies[1]) + len(record[0].harmonies[2]) for record in game_records) / len(
                game_records)

            game_lengths.append(avg_game_length)
            harmony_counts.append(avg_harmonies)

            # Train model on new data
            print(f"Training for {epochs_per_iteration} epochs...")
            policy_loss, value_loss = train(
                self.model,
                game_records,
                epochs=epochs_per_iteration,
                batch_size=batch_size,
                lr=lr,
                return_losses=True
            )

            # Record training metrics
            iteration_numbers.append(iteration + 1)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)

            # Calculate training time for this iteration
            iter_time = time.time() - iter_start_time
            training_times.append(iter_time)
            print(f"Iteration {iteration + 1} completed in {iter_time:.1f} seconds")
            print(f"Policy loss: {policy_loss:.4f}, Value loss: {value_loss:.4f}")
            print(f"Avg game length: {avg_game_length:.1f}, Avg harmonies: {avg_harmonies:.1f}")

            # Save model checkpoint
            if (iteration + 1) % save_interval == 0 or iteration == iterations - 1:
                self.save_model(f"transformer_model_iter_{iteration + 1}.pth")

        # Calculate total training time
        total_training_time = time.time() - start_time

        # Store metrics
        self.metrics["training"] = {
            "iterations": iterations,
            "games_per_iteration": games_per_iteration,
            "epochs_per_iteration": epochs_per_iteration,
            "batch_size": batch_size,
            "learning_rate": lr,
            "iteration_numbers": iteration_numbers,
            "policy_losses": policy_losses,
            "value_losses": value_losses,
            "game_lengths": game_lengths,
            "harmony_counts": harmony_counts,
            "training_times": training_times,
            "total_training_time": total_training_time
        }

        # Generate and save training plots
        self.generate_training_plots()

        print(f"\nTraining completed in {total_training_time:.1f} seconds")

        return self.metrics["training"]

    def generate_training_plots(self):
        """Generate and save plots for training metrics."""
        if "training" not in self.metrics:
            print("No training metrics available.")
            return

        metrics = self.metrics["training"]
        iteration_numbers = metrics["iteration_numbers"]

        # Create figure for plots
        plt.figure(figsize=(15, 12))

        # Plot policy loss
        plt.subplot(2, 2, 1)
        plt.plot(iteration_numbers, metrics["policy_losses"])
        plt.title("Policy Loss vs. Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Policy Loss")
        plt.grid(True)

        # Plot value loss
        plt.subplot(2, 2, 2)
        plt.plot(iteration_numbers, metrics["value_losses"])
        plt.title("Value Loss vs. Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Value Loss")
        plt.grid(True)

        # Plot game length
        plt.subplot(2, 2, 3)
        plt.plot(iteration_numbers, metrics["game_lengths"])
        plt.title("Average Game Length vs. Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Game Length (moves)")
        plt.grid(True)

        # Plot harmony count
        plt.subplot(2, 2, 4)
        plt.plot(iteration_numbers, metrics["harmony_counts"])
        plt.title("Average Harmony Count vs. Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Harmony Count")
        plt.grid(True)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "training_metrics.png"))
        plt.close()

        # Create a separate plot for training time
        plt.figure(figsize=(10, 6))
        plt.plot(iteration_numbers, metrics["training_times"])
        plt.title("Training Time per Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Time (seconds)")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "training_time.png"))
        plt.close()

    def evaluate_model(self, num_games=100, opponent="random"):
        """
        Evaluate the model's performance against a specific opponent.

        Args:
            num_games: Number of games to play for evaluation
            opponent: Type of opponent ("random", "self", or "greedy")
        """
        if self.model is None:
            print("Model must be initialized.")
            return None

        print(f"Evaluating model against {opponent} opponent over {num_games} games...")

        # Set model to evaluation mode
        self.model.eval()

        # Game statistics
        wins = 0
        losses = 0
        draws = 0
        game_lengths = []
        harmony_counts_model = []
        harmony_counts_opponent = []

        for game in range(num_games):
            state = SkudPaiShoState()
            game_length = 0

            # Randomly assign model to player 1 or 2
            model_player = np.random.choice([1, 2])

            while not state.is_game_over() and game_length < 200:  # Max 200 moves
                current_player = state.current_player

                if current_player == model_player:  # Model's turn
                    # Get model move
                    action, value = get_ai_move(self.model, state)
                else:  # Opponent's turn
                    if opponent == "random":
                        # Random opponent: choose a random valid move
                        valid_moves = state.get_valid_moves()
                        if not valid_moves:
                            break
                        action = valid_moves[np.random.randint(len(valid_moves))]

                    elif opponent == "self":
                        # Self-play: get another move from the model
                        action, _ = get_ai_move(self.model, state)

                    elif opponent == "greedy":
                        # Greedy opponent: choose move that creates the most harmonies
                        valid_moves = state.get_valid_moves()
                        if not valid_moves:
                            break

                        best_harmony_count = -1
                        best_move = valid_moves[0]

                        for move in valid_moves:
                            # Try move
                            test_state = state.copy()
                            test_state.make_move(move)

                            # Count harmonies
                            harmony_count = len(test_state.harmonies[current_player])

                            if harmony_count > best_harmony_count:
                                best_harmony_count = harmony_count
                                best_move = move

                        action = best_move

                if action is None:
                    break  # No valid moves

                # Make move
                state.make_move(action)
                game_length += 1

            # Record game result
            if state.is_game_over():
                winner = state.get_winner()
                if winner == model_player:
                    wins += 1
                elif winner == 0:  # Draw
                    draws += 1
                else:
                    losses += 1
            else:
                # Maximum moves reached without winner
                draws += 1

            # Record statistics
            game_lengths.append(game_length)
            harmony_counts_model.append(len(state.harmonies[model_player]))
            harmony_counts_opponent.append(len(state.harmonies[3 - model_player]))

            if (game + 1) % 10 == 0:
                print(f"Evaluated {game + 1}/{num_games} games")

        # Compile statistics
        win_rate = wins / num_games
        draw_rate = draws / num_games
        loss_rate = losses / num_games

        avg_game_length = sum(game_lengths) / len(game_lengths)
        avg_harmonies_model = sum(harmony_counts_model) / len(harmony_counts_model)
        avg_harmonies_opponent = sum(harmony_counts_opponent) / len(harmony_counts_opponent)

        stats = {
            "opponent": opponent,
            "num_games": num_games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "loss_rate": loss_rate,
            "avg_game_length": avg_game_length,
            "avg_harmonies_model": avg_harmonies_model,
            "avg_harmonies_opponent": avg_harmonies_opponent
        }

        # Store evaluation results
        if "evaluations" not in self.metrics:
            self.metrics["evaluations"] = {}

        self.metrics["evaluations"][opponent] = stats

        # Generate evaluation plots
        self.generate_evaluation_plots()

        print("\nEvaluation results:")
        print(f"Win rate: {win_rate:.3f} ({wins}/{num_games})")
        print(f"Draw rate: {draw_rate:.3f} ({draws}/{num_games})")
        print(f"Loss rate: {loss_rate:.3f} ({losses}/{num_games})")
        print(f"Average game length: {avg_game_length:.1f}")
        print(f"Average harmonies - Model: {avg_harmonies_model:.1f}, Opponent: {avg_harmonies_opponent:.1f}")

        return stats

    def generate_evaluation_plots(self):
        """Generate and save plots for evaluation results."""
        if "evaluations" not in self.metrics:
            print("No evaluation metrics available.")
            return

        # Create bar chart comparing win rates against different opponents
        plt.figure(figsize=(10, 6))

        opponents = list(self.metrics["evaluations"].keys())
        win_rates = [self.metrics["evaluations"][opponent]["win_rate"] for opponent in opponents]
        draw_rates = [self.metrics["evaluations"][opponent]["draw_rate"] for opponent in opponents]
        loss_rates = [self.metrics["evaluations"][opponent]["loss_rate"] for opponent in opponents]

        x = np.arange(len(opponents))
        width = 0.25

        plt.bar(x - width, win_rates, width, label='Wins')
        plt.bar(x, draw_rates, width, label='Draws')
        plt.bar(x + width, loss_rates, width, label='Losses')

        plt.xlabel('Opponent')
        plt.ylabel('Rate')
        plt.title('Performance Against Different Opponents')
        plt.xticks(x, opponents)
        plt.legend()
        plt.grid(True, axis='y')

        plt.savefig(os.path.join(self.results_dir, "evaluation_results.png"))
        plt.close()

    def architecture_study(self, architectures, iterations=20, games_per_iteration=10, epochs_per_iteration=3):
        """
        Study the impact of different model architectures.

        Args:
            architectures: List of dictionaries with architecture parameters
            iterations: Number of iterations for each architecture
            games_per_iteration: Number of self-play games per iteration
            epochs_per_iteration: Number of training epochs per iteration
        """
        print(f"Studying {len(architectures)} different architectures")

        results = []

        for i, arch_params in enumerate(architectures):
            print(f"\nArchitecture {i + 1}/{len(architectures)}:")
            print(f"Parameters: {arch_params}")

            # Initialize new model with these parameters
            self.initialize_model(**arch_params)

            # Train for specified number of iterations
            self.train_model(
                iterations=iterations,
                games_per_iteration=games_per_iteration,
                epochs_per_iteration=epochs_per_iteration,
                save_interval=iterations  # Only save at the end
            )

            # Evaluate performance
            eval_stats = self.evaluate_model(num_games=50, opponent="random")

            # Record results
            arch_results = {
                "parameters": arch_params,
                "win_rate": eval_stats["win_rate"],
                "avg_game_length": eval_stats["avg_game_length"],
                "avg_harmonies": eval_stats["avg_harmonies_model"],
                "final_policy_loss": self.metrics["training"]["policy_losses"][-1],
                "final_value_loss": self.metrics["training"]["value_losses"][-1],
                "training_time": sum(self.metrics["training"]["training_times"])
            }

            results.append(arch_results)

            # Save this architecture's model
            self.save_model(f"architecture_{i + 1}_model.pth")

        # Store architecture study results
        self.metrics["architecture_study"] = results

        # Generate architecture comparison plots
        self.generate_architecture_plots(results)

        return results

    def generate_architecture_plots(self, results):
        """Generate and save plots comparing different architectures."""
        if not results:
            print("No architecture study results available.")
            return

        # Create a summary plot
        plt.figure(figsize=(15, 10))

        # Sort architectures by win rate
        sorted_results = sorted(results, key=lambda x: x["win_rate"], reverse=True)

        # Extract parameter values for plotting
        arch_labels = []
        win_rates = []
        policy_losses = []
        value_losses = []
        training_times = []

        for i, result in enumerate(sorted_results):
            # Create a label from the parameter values
            label = f"A{i + 1}"
            arch_labels.append(label)

            win_rates.append(result["win_rate"])
            policy_losses.append(result["final_policy_loss"])
            value_losses.append(result["final_value_loss"])
            training_times.append(result["training_time"] / 60)  # Convert to minutes

        # Plot win rates
        plt.subplot(2, 2, 1)
        plt.bar(arch_labels, win_rates)
        plt.title("Win Rate by Architecture")
        plt.xlabel("Architecture")
        plt.ylabel("Win Rate")
        plt.grid(True, axis='y')

        # Plot policy loss
        plt.subplot(2, 2, 2)
        plt.bar(arch_labels, policy_losses)
        plt.title("Final Policy Loss by Architecture")
        plt.xlabel("Architecture")
        plt.ylabel("Policy Loss")
        plt.grid(True, axis='y')

        # Plot value loss
        plt.subplot(2, 2, 3)
        plt.bar(arch_labels, value_losses)
        plt.title("Final Value Loss by Architecture")
        plt.xlabel("Architecture")
        plt.ylabel("Value Loss")
        plt.grid(True, axis='y')

        # Plot training time
        plt.subplot(2, 2, 4)
        plt.bar(arch_labels, training_times)
        plt.title("Training Time by Architecture")
        plt.xlabel("Architecture")
        plt.ylabel("Time (minutes)")
        plt.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "architecture_comparison.png"))
        plt.close()

        # Create a table with architecture details
        plt.figure(figsize=(12, len(results) * 0.5 + 1))
        plt.axis('off')

        table_data = []
        table_data.append(["ID", "d_model", "nhead", "num_layers", "dropout", "Win Rate", "Training Time (min)"])

        for i, result in enumerate(sorted_results):
            params = result["parameters"]
            row = [
                f"A{i + 1}",
                params["d_model"],
                params["nhead"],
                params["num_layers"],
                params["dropout"],
                f"{result['win_rate']:.3f}",
                f"{result['training_time'] / 60:.1f}"
            ]
            table_data.append(row)

        table = plt.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.1] * 7)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title("Architecture Parameters and Performance", pad=20)

        plt.savefig(os.path.join(self.results_dir, "architecture_table.png"), bbox_inches='tight')
        plt.close()

    def policy_analysis(self, num_games=20):
        """
        Analyze the policy (move choices) of the transformer model.

        Args:
            num_games: Number of games to play for analysis
        """
        if self.model is None:
            print("Model must be initialized.")
            return None

        print(f"Analyzing model policy over {num_games} games...")

        # Set model to evaluation mode
        self.model.eval()

        # Count of each action type
        plant_actions = 0
        move_actions = 0

        # Count of tile types planted
        tile_types_planted = {}

        # Count of move distances
        move_distances = {}

        # Track entropy of policy distribution over time
        turn_entropy = defaultdict(list)

        # Play games for analysis
        for game in range(num_games):
            state = SkudPaiShoState()

            while not state.is_game_over() and state.turn_number < 200:
                # Get all valid moves
                valid_moves = state.get_valid_moves()

                if not valid_moves:
                    break

                # Convert valid moves to indices
                valid_indices = [action_to_index(move) for move in valid_moves]

                # Get model prediction
                state_tensor = torch.tensor(state.encode_for_network(), dtype=torch.float32).unsqueeze(0).to(
                    self.device)
                with torch.no_grad():
                    policy_logits, value = self.model(state_tensor)

                # Convert to probability distribution over valid moves
                policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

                # Filter for valid moves
                valid_policy = np.zeros_like(policy)
                for i, idx in enumerate(valid_indices):
                    valid_policy[idx] = policy[idx]

                # Normalize
                if valid_policy.sum() > 0:
                    valid_policy /= valid_policy.sum()

                # Calculate entropy of policy distribution
                policy_entropy = 0
                for p in valid_policy:
                    if p > 0:
                        policy_entropy -= p * np.log2(p)

                # Record entropy by turn number
                turn_entropy[state.turn_number].append(policy_entropy)

                # Choose best move
                action_idx = np.argmax(valid_policy)
                action = None

                for i, idx in enumerate(valid_indices):
                    if idx == action_idx:
                        action = valid_moves[i]
                        break

                if action is None:
                    action = valid_moves[0]  # Fallback

                # Analyze action
                if action[0] == "plant":
                    plant_actions += 1

                    # Record tile type
                    tile_type = action[1].name
                    tile_types_planted[tile_type] = tile_types_planted.get(tile_type, 0) + 1

                elif action[0] == "move":
                    move_actions += 1

                    # Calculate move distance
                    _, from_x, from_y, to_x, to_y = action
                    distance = int(((to_x - from_x) ** 2 + (to_y - from_y) ** 2) ** 0.5)
                    move_distances[distance] = move_distances.get(distance, 0) + 1

                # Make move
                state.make_move(action)

            if (game + 1) % 5 == 0:
                print(f"Analyzed {game + 1}/{num_games} games")

        # Calculate average entropy by turn
        avg_entropy_by_turn = {turn: sum(entropies) / len(entropies) for turn, entropies in turn_entropy.items()}

        # Compile statistics
        stats = {
            'total_actions': plant_actions + move_actions,
            'plant_actions': plant_actions,
            'move_actions': move_actions,
            'plant_percentage': plant_actions / (plant_actions + move_actions) * 100 if (
                                                                                                    plant_actions + move_actions) > 0 else 0,
            'move_percentage': move_actions / (plant_actions + move_actions) * 100 if (
                                                                                                  plant_actions + move_actions) > 0 else 0,
            'tile_types_planted': tile_types_planted,
            'move_distances': move_distances,
            'avg_entropy_by_turn': avg_entropy_by_turn
        }

        # Store results
        self.metrics["policy_analysis"] = stats

        # Generate policy analysis plots
        self.generate_policy_plots()

        return stats

    def generate_policy_plots(self):
        """Generate and save plots for policy analysis."""
        if "policy_analysis" not in self.metrics:
            print("No policy analysis available.")
            return

        stats = self.metrics["policy_analysis"]

        # Plot action type distribution
        plt.figure(figsize=(10, 6))
        labels = ['Plant', 'Move']
        values = [stats['plant_percentage'], stats['move_percentage']]

        plt.bar(labels, values)
        plt.title("Action Type Distribution")
        plt.ylabel("Percentage (%)")
        plt.ylim(0, 100)

        for i, v in enumerate(values):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center')

        plt.savefig(os.path.join(self.results_dir, "action_types.png"))
        plt.close()

        # Plot tile types planted
        if stats['tile_types_planted']:
            plt.figure(figsize=(12, 6))

            labels = list(stats['tile_types_planted'].keys())
            values = list(stats['tile_types_planted'].values())

            # Sort by frequency
            sorted_indices = np.argsort(values)[::-1]
            labels = [labels[i] for i in sorted_indices]
            values = [values[i] for i in sorted_indices]

            plt.bar(labels, values)
            plt.title("Tile Types Planted")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            plt.savefig(os.path.join(self.results_dir, "tile_types_planted.png"))
            plt.close()

        # Plot move distances
        if stats['move_distances']:
            plt.figure(figsize=(10, 6))

            distances = sorted(stats['move_distances'].keys())
            counts = [stats['move_distances'][d] for d in distances]

            plt.bar(distances, counts)
            plt.title("Move Distances")
            plt.xlabel("Distance")
            plt.ylabel("Count")

            plt.savefig(os.path.join(self.results_dir, "move_distances.png"))
            plt.close()

        # Plot policy entropy by turn
        if stats['avg_entropy_by_turn']:
            plt.figure(figsize=(10, 6))

            turns = sorted(stats['avg_entropy_by_turn'].keys())
            entropies = [stats['avg_entropy_by_turn'][t] for t in turns]

            plt.plot(turns, entropies, 'o-')
            plt.title("Policy Entropy by Turn Number")
            plt.xlabel("Turn")
            plt.ylabel("Average Entropy (bits)")
            plt.grid(True)

            plt.savefig(os.path.join(self.results_dir, "policy_entropy.png"))
            plt.close()

    def attention_analysis(self, num_samples=10):
        """
        Analyze the attention patterns in the transformer.

        Args:
            num_samples: Number of game states to analyze
        """
        if self.model is None:
            print("Model must be initialized.")
            return None

        print(f"Analyzing attention patterns over {num_samples} game states...")

        # Set model to evaluation mode
        self.model.eval()

        # Create a small dataset of game states
        states = []
        state_descriptions = []

        # Generate states at different game phases
        state = SkudPaiShoState()
        states.append(state.copy())
        state_descriptions.append("Initial state")

        # Play a game and sample states
        moves_played = 0
        target_moves = [0, 2, 5, 10, 15, 20, 30, 40, 50, 60]  # Sample at these move counts

        while not state.is_game_over() and moves_played < max(target_moves):
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                break

            # Choose a move (using model)
            action, _ = get_ai_move(self.model, state)

            if action is None:
                break

            # Make move
            state.make_move(action)
            moves_played += 1

            # Check if we should sample this state
            if moves_played in target_moves:
                states.append(state.copy())
                state_descriptions.append(f"After {moves_played} moves")

        # Limit to requested number of samples
        states = states[:num_samples]
        state_descriptions = state_descriptions[:num_samples]

        # For visualization, we'll focus on the first layer attention
        # Since full attention analysis is complex, we'll generate heatmaps of board positions
        attention_scores = []

        for i, state in enumerate(states):
            print(f"Analyzing state {i + 1}/{len(states)}")

            # Get state representation
            state_tensor = torch.tensor(state.encode_for_network(), dtype=torch.float32).unsqueeze(0).to(self.device)

            # Forward pass through the model
            with torch.no_grad():
                # Get first layer attention scores (simplified approximation)
                # In a real implementation, you'd use hooks to extract actual attention weights
                x = self.model.conv1(state_tensor)  # Initial conv
                attention_score = torch.mean(x, dim=1).squeeze(0).cpu().numpy()

                # Reshape to match board
                attention_score = attention_score.reshape(BOARD_SIZE, BOARD_SIZE)
                attention_scores.append(attention_score)

        # Store results
        self.metrics["attention_analysis"] = {
            "state_descriptions": state_descriptions,
            "attention_scores": attention_scores
        }

        # Generate attention heatmap plots
        self.generate_attention_plots()

        return attention_scores

    def generate_attention_plots(self):
        """Generate and save attention heatmap plots."""
        if "attention_analysis" not in self.metrics:
            print("No attention analysis available.")
            return

        metrics = self.metrics["attention_analysis"]
        state_descriptions = metrics["state_descriptions"]
        attention_scores = metrics["attention_scores"]

        # Create a directory for attention maps
        attention_dir = os.path.join(self.results_dir, "attention_maps")
        os.makedirs(attention_dir, exist_ok=True)

        # Generate a heatmap for each state
        for i, (description, attention) in enumerate(zip(state_descriptions, attention_scores)):
            plt.figure(figsize=(10, 8))

            # Create a masked version for the circular board
            mask = np.zeros_like(attention, dtype=bool)

            # Create circular mask
            center = BOARD_SIZE // 2
            for y in range(BOARD_SIZE):
                for x in range(BOARD_SIZE):
                    # Distance from center
                    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
                    if dist > BOARD_RADIUS:
                        mask[y, x] = True

            # Plot heatmap
            plt.imshow(np.ma.masked_array(attention, mask), cmap='viridis')
            plt.colorbar(label='Attention Score')
            plt.title(f"Attention Map: {description}")

            # Save figure
            plt.savefig(os.path.join(attention_dir, f"attention_map_{i + 1}.png"))
            plt.close()

        # Create a combined figure showing progression
        num_maps = min(4, len(attention_scores))  # Show at most 4 maps in combined figure

        plt.figure(figsize=(15, 10))

        for i in range(num_maps):
            plt.subplot(2, 2, i + 1)

            # Create a masked version for the circular board
            idx = i * (len(attention_scores) // num_maps)
            attention = attention_scores[idx]
            description = state_descriptions[idx]

            mask = np.zeros_like(attention, dtype=bool)

            # Create circular mask
            center = BOARD_SIZE // 2
            for y in range(BOARD_SIZE):
                for x in range(BOARD_SIZE):
                    # Distance from center
                    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
                    if dist > BOARD_RADIUS:
                        mask[y, x] = True

            # Plot heatmap
            plt.imshow(np.ma.masked_array(attention, mask), cmap='viridis')
            plt.colorbar(label='Attention Score')
            plt.title(description)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "attention_progression.png"))
        plt.close()

    def training_efficiency_study(self, training_configs):
        """
        Study the training efficiency of different configurations.

        Args:
            training_configs: List of dictionaries with training parameters
        """
        print(f"Studying {len(training_configs)} different training configurations")

        results = []

        for i, config in enumerate(training_configs):
            print(f"\nTraining configuration {i + 1}/{len(training_configs)}:")
            print(f"Parameters: {config}")

            # Initialize a new model for each config
            self.initialize_model()

            # Train with these parameters
            start_time = time.time()

            self.train_model(
                iterations=config["iterations"],
                games_per_iteration=config["games_per_iteration"],
                epochs_per_iteration=config["epochs_per_iteration"],
                batch_size=config.get("batch_size", 128),
                lr=config.get("lr", 0.001),
                save_interval=config["iterations"]  # Only save at the end
            )

            # Calculate training time
            training_time = time.time() - start_time

            # Evaluate performance
            eval_stats = self.evaluate_model(num_games=50, opponent="random")

            # Record results
            config_results = {
                "parameters": config,
                "win_rate": eval_stats["win_rate"],
                "training_time": training_time,
                "training_time_minutes": training_time / 60,
                "iterations": config["iterations"],
                "total_games": config["iterations"] * config["games_per_iteration"],
                "total_epochs": config["iterations"] * config["epochs_per_iteration"],
                "final_policy_loss": self.metrics["training"]["policy_losses"][-1],
                "final_value_loss": self.metrics["training"]["value_losses"][-1],
            }

            results.append(config_results)

            # Save this config's model
            self.save_model(f"training_config_{i + 1}_model.pth")

        # Store training efficiency study results
        self.metrics["training_efficiency_study"] = results

        # Generate training efficiency comparison plots
        self.generate_efficiency_plots(results)

        return results

    def generate_efficiency_plots(self, results):
        """Generate and save plots comparing training efficiency."""
        if not results:
            print("No training efficiency study results available.")
            return

        # Create efficiency plots
        plt.figure(figsize=(15, 10))

        # Sort configurations by win rate
        sorted_results = sorted(results, key=lambda x: x["win_rate"], reverse=True)

        # Extract data for plotting
        config_labels = [f"C{i + 1}" for i in range(len(sorted_results))]
        win_rates = [r["win_rate"] for r in sorted_results]
        training_times = [r["training_time_minutes"] for r in sorted_results]

        # Calculate efficiency metric (win rate / training time)
        efficiency = [w / max(t, 0.1) for w, t in zip(win_rates, training_times)]

        # Plot win rates
        plt.subplot(2, 2, 1)
        plt.bar(config_labels, win_rates)
        plt.title("Win Rate by Configuration")
        plt.xlabel("Configuration")
        plt.ylabel("Win Rate")
        plt.grid(True, axis='y')

        # Plot training times
        plt.subplot(2, 2, 2)
        plt.bar(config_labels, training_times)
        plt.title("Training Time by Configuration")
        plt.xlabel("Configuration")
        plt.ylabel("Time (minutes)")
        plt.grid(True, axis='y')

        # Plot efficiency
        plt.subplot(2, 2, 3)
        plt.bar(config_labels, efficiency)
        plt.title("Efficiency (Win Rate / Training Time)")
        plt.xlabel("Configuration")
        plt.ylabel("Efficiency")
        plt.grid(True, axis='y')

        # Plot total games
        plt.subplot(2, 2, 4)
        total_games = [r["total_games"] for r in sorted_results]
        plt.bar(config_labels, total_games)
        plt.title("Total Self-Play Games")
        plt.xlabel("Configuration")
        plt.ylabel("Number of Games")
        plt.grid(True, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "training_efficiency.png"))
        plt.close()

        # Create a table with configuration details
        plt.figure(figsize=(12, len(results) * 0.5 + 1))
        plt.axis('off')

        table_data = []
        table_data.append(["ID", "Iterations", "Games/Iter", "Epochs/Iter", "Win Rate", "Time (min)", "Efficiency"])

        for i, result in enumerate(sorted_results):
            params = result["parameters"]
            row = [
                f"C{i + 1}",
                params["iterations"],
                params["games_per_iteration"],
                params["epochs_per_iteration"],
                f"{result['win_rate']:.3f}",
                f"{result['training_time_minutes']:.1f}",
                f"{efficiency[i]:.4f}"
            ]
            table_data.append(row)

        table = plt.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.1] * 7)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.title("Training Configuration Parameters and Performance", pad=20)

        plt.savefig(os.path.join(self.results_dir, "training_config_table.png"), bbox_inches='tight')
        plt.close()

    def compare_with_q_learning(self, q_agent_path, num_games=100):
        """
        Compare transformer performance with a Q-learning agent.

        Args:
            q_agent_path: Path to the saved Q-learning agent
            num_games: Number of games to play for comparison
        """
        if self.model is None:
            print("Transformer model must be initialized.")
            return None

        # Load Q-learning agent
        from q_learning import SkudPaiShoQLearning
        q_agent = SkudPaiShoQLearning()
        q_agent.load(q_agent_path)

        print(f"Comparing transformer with Q-learning over {num_games} games...")

        # Game statistics
        transformer_wins = 0
        q_learning_wins = 0
        draws = 0
        game_lengths = []

        for game in range(num_games):
            state = SkudPaiShoState()
            game_length = 0

            # Randomly determine which agent plays as player 1
            transformer_player = np.random.choice([1, 2])
            q_learning_player = 3 - transformer_player

            while not state.is_game_over() and game_length < 200:
                current_player = state.current_player

                if current_player == transformer_player:
                    # Transformer's turn
                    action, _ = get_ai_move(self.model, state)
                else:
                    # Q-learning's turn
                    action = q_agent.choose_action(state)

                if action is None:
                    break

                # Make move
                state.make_move(action)
                game_length += 1

            # Record game result
            if state.is_game_over():
                winner = state.get_winner()
                if winner == transformer_player:
                    transformer_wins += 1
                elif winner == q_learning_player:
                    q_learning_wins += 1
                else:
                    draws += 1
            else:
                # Maximum moves reached
                draws += 1

            # Record game length
            game_lengths.append(game_length)

            if (game + 1) % 10 == 0:
                print(f"Completed {game + 1}/{num_games} games")

        # Compile statistics
        transformer_win_rate = transformer_wins / num_games
        q_learning_win_rate = q_learning_wins / num_games
        draw_rate = draws / num_games
        avg_game_length = sum(game_lengths) / len(game_lengths)

        stats = {
            "num_games": num_games,
            "transformer_wins": transformer_wins,
            "q_learning_wins": q_learning_wins,
            "draws": draws,
            "transformer_win_rate": transformer_win_rate,
            "q_learning_win_rate": q_learning_win_rate,
            "draw_rate": draw_rate,
            "avg_game_length": avg_game_length
        }

        # Store comparison results
        self.metrics["q_learning_comparison"] = stats

        # Generate comparison plot
        self.generate_comparison_plot()

        print("\nComparison results:")
        print(f"Transformer wins: {transformer_wins}/{num_games} ({transformer_win_rate:.3f})")
        print(f"Q-learning wins: {q_learning_wins}/{num_games} ({q_learning_win_rate:.3f})")
        print(f"Draws: {draws}/{num_games} ({draw_rate:.3f})")
        print(f"Average game length: {avg_game_length:.1f}")

        return stats

    def generate_comparison_plot(self):
        """Generate and save a plot comparing transformer with Q-learning."""
        if "q_learning_comparison" not in self.metrics:
            print("No Q-learning comparison available.")
            return

        stats = self.metrics["q_learning_comparison"]

        # Create bar chart of win rates
        plt.figure(figsize=(10, 6))

        labels = ['Transformer', 'Q-Learning', 'Draw']
        values = [stats['transformer_win_rate'], stats['q_learning_win_rate'], stats['draw_rate']]
        colors = ['blue', 'green', 'gray']

        plt.bar(labels, values, color=colors)
        plt.title("Win Rates: Transformer vs. Q-Learning")
        plt.ylabel("Win Rate")
        plt.ylim(0, 1.0)

        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center')

        plt.savefig(os.path.join(self.results_dir, "transformer_vs_qlearning.png"))
        plt.close()

    def save_study_results(self):
        """Save all study results to JSON file."""

        # Convert all numpy arrays and non-serializable objects to lists
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        # Use the conversion function on the entire metrics dictionary
        serializable_metrics = convert_to_serializable(self.metrics)
        serializable_parameters = convert_to_serializable(self.parameters)

        # Save to file
        with open(os.path.join(self.results_dir, "study_results.json"), 'w') as f:
            json.dump({
                "study_name": self.study_name,
                "parameters": serializable_parameters,
                "metrics": serializable_metrics
            }, f, indent=2)

        print(f"Study results saved to {os.path.join(self.results_dir, 'study_results.json')}")

    def load_study_results(self, study_name=None):
        """Load study results from JSON file."""
        if study_name is not None:
            self.study_name = study_name
            self.results_dir = os.path.join("study_results", study_name)

        try:
            with open(os.path.join(self.results_dir, "study_results.json"), 'r') as f:
                data = json.load(f)

                self.study_name = data["study_name"]
                self.parameters = data["parameters"]
                self.metrics = data["metrics"]

                print(f"Study results loaded from {os.path.join(self.results_dir, 'study_results.json')}")
                return True
        except:
            print(f"Could not load study results from {os.path.join(self.results_dir, 'study_results.json')}")
            return False

    def run_full_study(self, iterations=50, games_per_iteration=20, arch_study=True, efficiency_study=True,
                       compare_q=True):
        """Run a comprehensive study of the transformer model for Skud Pai Sho."""
        print(f"Starting comprehensive transformer study: {self.study_name}")

        # 1. Train basic transformer model
        print("\n1. Training basic transformer model...")
        self.initialize_model()
        self.train_model(
            iterations=iterations,
            games_per_iteration=games_per_iteration
        )

        # 2. Evaluate the model against different opponents
        print("\n2. Evaluating model against different opponents...")
        for opponent in ["random", "greedy", "self"]:
            self.evaluate_model(num_games=50, opponent=opponent)

        # 3. Perform policy analysis
        print("\n3. Performing policy analysis...")
        self.policy_analysis(num_games=20)

        # 4. Perform attention analysis
        print("\n4. Performing attention analysis...")
        self.attention_analysis(num_samples=8)

        # 5. Architecture study (if enabled)
        if arch_study:
            print("\n5. Performing architecture study...")
            architectures = [
                {"d_model": 128, "nhead": 4, "num_layers": 3, "dropout": 0.1},
                {"d_model": 256, "nhead": 8, "num_layers": 6, "dropout": 0.1},
                {"d_model": 512, "nhead": 8, "num_layers": 6, "dropout": 0.1},
                {"d_model": 256, "nhead": 4, "num_layers": 9, "dropout": 0.1},
                {"d_model": 256, "nhead": 8, "num_layers": 3, "dropout": 0.2}
            ]
            self.architecture_study(architectures, iterations=10, games_per_iteration=10)

        # 6. Training efficiency study (if enabled)
        if efficiency_study:
            print("\n6. Performing training efficiency study...")
            training_configs = [
                {"iterations": 20, "games_per_iteration": 10, "epochs_per_iteration": 3},
                {"iterations": 10, "games_per_iteration": 20, "epochs_per_iteration": 3},
                {"iterations": 5, "games_per_iteration": 40, "epochs_per_iteration": 3},
                {"iterations": 20, "games_per_iteration": 10, "epochs_per_iteration": 6},
                {"iterations": 40, "games_per_iteration": 5, "epochs_per_iteration": 3}
            ]
            self.training_efficiency_study(training_configs)

        # 7. Compare with Q-learning (if enabled)
        if compare_q:
            try:
                q_agent_path = input("Enter path to Q-learning agent (or press Enter to skip): ")
                if q_agent_path:
                    print("\n7. Comparing with Q-learning agent...")
                    self.compare_with_q_learning(q_agent_path)
            except:
                print("Skipping Q-learning comparison (error loading agent).")

        # 8. Save all results
        print("\n8. Saving study results...")
        self.save_study_results()

        print(f"\nComprehensive transformer study completed: {self.study_name}")
        return self.metrics