# study_framework.py
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json
from q_learning_ai import SkudPaiShoQLearning
from game.state import SkudPaiShoState
from game.display import display_board, move_to_string
from ai_deprecated.model import SkudPaiShoTransformer
from ai_deprecated.utils import create_initial_model, get_ai_move


class SkudPaiShoStudy:
    def __init__(self, study_name="skud_pai_sho_study"):
        """Initialize the study framework."""
        self.study_name = study_name
        os.makedirs("study_results", exist_ok=True)
        self.results_dir = os.path.join("study_results", study_name)
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize agents
        self.q_agent = None
        self.transformer_model = None

        # Study parameters
        self.parameters = {}
        self.metrics = {}

    def initialize_q_agent(self, learning_rate=0.1, discount_factor=0.9,
                           exploration_rate=1.0, exploration_decay=0.995, min_exploration_rate=0.01):
        """Initialize the Q-learning agent with specific parameters."""
        self.q_agent = SkudPaiShoQLearning(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate,
            exploration_decay=exploration_decay,
            min_exploration_rate=min_exploration_rate
        )

        self.parameters["q_learning"] = {
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "exploration_rate": exploration_rate,
            "exploration_decay": exploration_decay,
            "min_exploration_rate": min_exploration_rate
        }

        return self.q_agent

    def initialize_transformer(self):
        """Initialize the transformer model."""
        self.transformer_model = create_initial_model()
        return self.transformer_model

    def load_q_agent(self, filename):
        """Load a pre-trained Q-learning agent."""
        self.q_agent = SkudPaiShoQLearning()
        self.q_agent.load(filename)
        return self.q_agent

    def load_transformer(self, filename):
        """Load a pre-trained transformer model."""
        self.transformer_model = create_initial_model()
        self.transformer_model.load_state_dict(torch.load(filename))
        return self.transformer_model

    def train_q_agent(self, num_episodes=1000, print_interval=100):
        """Train the Q-learning agent and record performance metrics."""
        if self.q_agent is None:
            self.initialize_q_agent()

        # Record training start time
        start_time = time.time()

        # Initialize metrics tracking
        episode_numbers = []
        win_rates = []
        exploration_rates = []
        q_table_sizes = []
        avg_game_lengths = []

        # Train in batches and record metrics
        batch_size = print_interval
        for start_episode in range(1, num_episodes + 1, batch_size):
            end_episode = min(start_episode + batch_size - 1, num_episodes)
            batch_episodes = end_episode - start_episode + 1

            # Train for batch_episodes
            for _ in range(batch_episodes):
                self.q_agent.train_episode()

            # Record metrics
            episode_numbers.append(end_episode)
            win_rates.append(self.q_agent.wins / self.q_agent.games_played)
            exploration_rates.append(self.q_agent.exploration_rate)
            q_table_sizes.append(len(self.q_agent.q_table))
            avg_game_lengths.append(self.q_agent.avg_game_length)

            # Print progress
            print(f"Episodes {start_episode}-{end_episode}/{num_episodes} completed")
            print(f"Win rate: {win_rates[-1]:.3f}, Exploration rate: {exploration_rates[-1]:.3f}")
            print(f"Q-table size: {q_table_sizes[-1]}, Avg game length: {avg_game_lengths[-1]:.1f}")
            print("-" * 50)

        # Calculate training time
        training_time = time.time() - start_time

        # Store metrics
        self.metrics["q_learning_training"] = {
            "num_episodes": num_episodes,
            "episode_numbers": episode_numbers,
            "win_rates": win_rates,
            "exploration_rates": exploration_rates,
            "q_table_sizes": q_table_sizes,
            "avg_game_lengths": avg_game_lengths,
            "training_time": training_time
        }

        # Save the agent
        model_path = os.path.join(self.results_dir, "q_agent.pkl")
        self.q_agent.save(model_path)

        # Generate and save plots
        self.generate_training_plots()

        return self.metrics["q_learning_training"]

    def generate_training_plots(self):
        """Generate and save plots for training metrics."""
        if "q_learning_training" not in self.metrics:
            print("No training metrics available.")
            return

        metrics = self.metrics["q_learning_training"]
        episode_numbers = metrics["episode_numbers"]

        # Create figure for plots
        plt.figure(figsize=(15, 12))

        # Plot win rate
        plt.subplot(2, 2, 1)
        plt.plot(episode_numbers, metrics["win_rates"])
        plt.title("Win Rate vs. Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Win Rate")
        plt.grid(True)

        # Plot exploration rate
        plt.subplot(2, 2, 2)
        plt.plot(episode_numbers, metrics["exploration_rates"])
        plt.title("Exploration Rate vs. Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Exploration Rate")
        plt.grid(True)

        # Plot Q-table size
        plt.subplot(2, 2, 3)
        plt.plot(episode_numbers, metrics["q_table_sizes"])
        plt.title("Q-table Size vs. Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Number of State-Action Pairs")
        plt.grid(True)

        # Plot average game length
        plt.subplot(2, 2, 4)
        plt.plot(episode_numbers, metrics["avg_game_lengths"])
        plt.title("Average Game Length vs. Episodes")
        plt.xlabel("Episodes")
        plt.ylabel("Average Game Length (moves)")
        plt.grid(True)

        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "training_metrics.png"))
        plt.close()

    def compare_q_vs_transformer(self, num_games=100):
        """Compare the performance of Q-learning agent against the transformer model."""
        if self.q_agent is None or self.transformer_model is None:
            print("Both Q-learning agent and transformer model must be initialized.")
            return None

        print(f"Playing {num_games} games: Q-learning vs. Transformer...")

        # Set exploration rate to minimum for evaluation
        old_exploration_rate = self.q_agent.exploration_rate
        self.q_agent.exploration_rate = self.q_agent.min_exploration_rate

        # Game statistics
        q_agent_wins = 0
        transformer_wins = 0
        draws = 0
        game_lengths = []
        harmony_counts_q = []
        harmony_counts_transformer = []

        for game in range(num_games):
            state = SkudPaiShoState()
            game_length = 0

            while not state.is_game_over() and game_length < 200:
                if state.current_player == 1:  # Q-agent's turn
                    action = self.q_agent.choose_action(state)
                else:  # Transformer's turn
                    action, _ = get_ai_move(self.transformer_model, state)

                if action is None:
                    break

                state.make_move(action)
                game_length += 1

            # Record statistics
            winner = state.get_winner()
            if winner == 1:  # Q-agent wins
                q_agent_wins += 1
            elif winner == 2:  # Transformer wins
                transformer_wins += 1
            else:
                draws += 1

            game_lengths.append(game_length)
            harmony_counts_q.append(len(state.harmonies[1]))
            harmony_counts_transformer.append(len(state.harmonies[2]))

            if (game + 1) % 10 == 0:
                print(f"Completed {game + 1}/{num_games} games...")

        # Restore exploration rate
        self.q_agent.exploration_rate = old_exploration_rate

        # Compile statistics
        stats = {
            'q_agent_wins': q_agent_wins,
            'transformer_wins': transformer_wins,
            'draws': draws,
            'q_agent_win_rate': q_agent_wins / num_games,
            'transformer_win_rate': transformer_wins / num_games,
            'draw_rate': draws / num_games,
            'avg_game_length': sum(game_lengths) / len(game_lengths),
            'avg_harmonies_q': sum(harmony_counts_q) / len(harmony_counts_q),
            'avg_harmonies_transformer': sum(harmony_counts_transformer) / len(harmony_counts_transformer)
        }

        # Save statistics
        self.metrics["comparison"] = stats

        # Print results
        print("\nComparison Results:")
        print(f"Q-learning wins: {q_agent_wins}/{num_games} ({stats['q_agent_win_rate']:.3f})")
        print(f"Transformer wins: {transformer_wins}/{num_games} ({stats['transformer_win_rate']:.3f})")
        print(f"Draws: {draws}/{num_games} ({stats['draw_rate']:.3f})")
        print(f"Average game length: {stats['avg_game_length']:.1f}")
        print(
            f"Average harmonies - Q-agent: {stats['avg_harmonies_q']:.1f}, Transformer: {stats['avg_harmonies_transformer']:.1f}")

        # Generate comparison plot
        self.generate_comparison_plot()

        return stats

    def generate_comparison_plot(self):
        """Generate and save a plot comparing Q-learning and transformer performance."""
        if "comparison" not in self.metrics:
            print("No comparison metrics available.")
            return

        stats = self.metrics["comparison"]

        # Create bar chart of win rates
        plt.figure(figsize=(10, 6))

        labels = ['Q-Learning', 'Transformer', 'Draw']
        values = [stats['q_agent_win_rate'], stats['transformer_win_rate'], stats['draw_rate']]
        colors = ['blue', 'green', 'gray']

        plt.bar(labels, values, color=colors)
        plt.title("Win Rates: Q-Learning vs. Transformer")
        plt.xlabel("Agent")
        plt.ylabel("Win Rate")
        plt.ylim(0, 1.0)

        for i, v in enumerate(values):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center')

        plt.savefig(os.path.join(self.results_dir, "comparison.png"))
        plt.close()

    def parameter_study(self, parameter_combinations, episodes_per_combo=1000):
        """
        Study the effect of different parameter combinations on Q-learning performance.

        Args:
            parameter_combinations: List of dictionaries with parameter combinations
            episodes_per_combo: Number of episodes to train for each combination
        """
        results = []

        for i, params in enumerate(parameter_combinations):
            print(f"Testing parameter combination {i + 1}/{len(parameter_combinations)}:")
            print(f"Parameters: {params}")

            # Initialize a new Q-agent with these parameters
            self.initialize_q_agent(**params)

            # Train for specified number of episodes
            self.q_agent.train(num_episodes=episodes_per_combo, print_interval=episodes_per_combo // 10)

            # Evaluate performance
            eval_stats = self.q_agent.play_against_self(num_games=100)

            # Record results
            combo_results = {
                "parameters": params,
                "win_rate_p1": eval_stats["win_rate_p1"],
                "win_rate_p2": eval_stats["win_rate_p2"],
                "draw_rate": eval_stats["draw_rate"],
                "avg_game_length": eval_stats["avg_game_length"],
                "q_table_size": len(self.q_agent.q_table)
            }

            results.append(combo_results)

            # Save the best performing agent
            if i == 0 or combo_results["win_rate_p1"] > best_win_rate:
                best_win_rate = combo_results["win_rate_p1"]
                self.q_agent.save(os.path.join(self.results_dir, "best_q_agent.pkl"))

        # Store results
        self.metrics["parameter_study"] = results

        # Generate parameter study plots
        self.generate_parameter_study_plots(results)

        return results

    def generate_parameter_study_plots(self, results):
        """Generate and save plots showing the effect of different parameters."""
        if not results:
            print("No parameter study results available.")
            return

        # Extract parameter names (assuming all dictionaries have the same keys)
        param_names = list(results[0]["parameters"].keys())

        # For each parameter, create a plot showing its effect on win rate
        for param_name in param_names:
            # Extract unique parameter values
            param_values = sorted(list(set(result["parameters"][param_name] for result in results)))

            if len(param_values) < 2:
                continue  # Skip parameters that don't vary

            # Group results by parameter value
            grouped_results = {}
            for value in param_values:
                grouped_results[value] = [r for r in results if r["parameters"][param_name] == value]

            # Calculate average win rates for each parameter value
            avg_win_rates = []
            for value in param_values:
                avg_win_rate = sum(r["win_rate_p1"] for r in grouped_results[value]) / len(grouped_results[value])
                avg_win_rates.append(avg_win_rate)

            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(param_values, avg_win_rates, 'o-')
            plt.title(f"Effect of {param_name} on Win Rate")
            plt.xlabel(param_name)
            plt.ylabel("Average Win Rate")
            plt.grid(True)

            plt.savefig(os.path.join(self.results_dir, f"param_study_{param_name}.png"))
            plt.close()

        # Create summary plot
        plt.figure(figsize=(12, 8))

        # Sort results by win rate
        sorted_results = sorted(results, key=lambda r: r["win_rate_p1"], reverse=True)

        # Take top 5 parameter combinations
        top_results = sorted_results[:5]

        x = list(range(len(top_results)))
        win_rates = [r["win_rate_p1"] for r in top_results]

        # Create labels from parameter combinations
        param_labels = []
        for result in top_results:
            label = ", ".join([f"{k}={v:.3f}" for k, v in result["parameters"].items()])
            param_labels.append(label)

        plt.bar(x, win_rates)
        plt.xticks(x, param_labels, rotation=45, ha="right")
        plt.title("Top 5 Parameter Combinations by Win Rate")
        plt.ylabel("Win Rate")
        plt.tight_layout()

        plt.savefig(os.path.join(self.results_dir, "top_params.png"))
        plt.close()

    def state_space_analysis(self, num_games=100):
        """
        Analyze the state space explored by the Q-learning agent.

        Args:
            num_games: Number of games to play for analysis
        """
        if self.q_agent is None:
            print("Q-learning agent must be initialized.")
            return None

        print(f"Analyzing state space over {num_games} games...")

        state_visits = {}  # Count of visits to each state
        action_distribution = {}  # Q-values for most commonly visited states
        game_progression = []  # Track state space growth over time

        # Record initial Q-table size
        game_progression.append(len(self.q_agent.q_table))

        # Play games for analysis
        for game in range(num_games):
            state = SkudPaiShoState()

            while not state.is_game_over() and state.turn_number < 200:
                # Record state
                state_key = self.q_agent.get_state_key(state)
                state_visits[state_key] = state_visits.get(state_key, 0) + 1

                # For commonly visited states, record action distribution
                if state_visits[state_key] >= 5:  # Record only for states visited at least 5 times
                    if state_key not in action_distribution:
                        action_distribution[state_key] = self.q_agent.get_action_distribution(state)

                # Choose action
                action = self.q_agent.choose_action(state)

                if action is None:
                    break

                # Make move
                state.make_move(action)

            # Record Q-table size after each game
            game_progression.append(len(self.q_agent.q_table))

            if (game + 1) % 10 == 0:
                print(f"Analyzed {game + 1}/{num_games} games...")

        # Compile statistics
        top_states = sorted(state_visits.items(), key=lambda x: x[1], reverse=True)[:10]

        stats = {
            'total_states_visited': len(state_visits),
            'q_table_size': len(self.q_agent.q_table),
            'top_visited_states': [(str(state), count) for state, count in top_states],
            'state_space_growth': game_progression
        }

        # Store results
        self.metrics["state_space_analysis"] = stats

        # Generate state space analysis plots
        self.generate_state_space_plots()

        return stats

    def generate_state_space_plots(self):
        """Generate and save plots for state space analysis."""
        if "state_space_analysis" not in self.metrics:
            print("No state space analysis available.")
            return

        stats = self.metrics["state_space_analysis"]

        # Plot state space growth
        plt.figure(figsize=(10, 6))
        plt.plot(stats["state_space_growth"])
        plt.title("Q-table Size vs. Games Played")
        plt.xlabel("Games")
        plt.ylabel("Number of State-Action Pairs")
        plt.grid(True)

        plt.savefig(os.path.join(self.results_dir, "state_space_growth.png"))
        plt.close()

    def action_analysis(self, num_games=20):
        """
        Analyze the actions chosen by the Q-learning agent.

        Args:
            num_games: Number of games to play for analysis
        """
        if self.q_agent is None:
            print("Q-learning agent must be initialized.")
            return None

        print(f"Analyzing action choices over {num_games} games...")

        # Count of each action type
        plant_actions = 0
        move_actions = 0

        # Count of tile types planted
        tile_types_planted = {}

        # Count of move distances
        move_distances = {}

        # Play games for analysis
        for game in range(num_games):
            state = SkudPaiShoState()

            while not state.is_game_over() and state.turn_number < 200:
                # Choose action
                action = self.q_agent.choose_action(state)

                if action is None:
                    break

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
                print(f"Analyzed {game + 1}/{num_games} games...")

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
            'move_distances': move_distances
        }

        # Store results
        self.metrics["action_analysis"] = stats

        # Generate action analysis plots
        self.generate_action_plots()

        return stats

    def generate_action_plots(self):
        """Generate and save plots for action analysis."""
        if "action_analysis" not in self.metrics:
            print("No action analysis available.")
            return

        stats = self.metrics["action_analysis"]

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

    def save_study_results(self):
        """Save all study results to JSON file."""
        # Convert all numpy arrays and other non-serializable objects to lists
        serializable_metrics = {}

        for key, value in self.metrics.items():
            if isinstance(value, dict):
                serializable_metrics[key] = {}
                for k, v in value.items():
                    if isinstance(v, (np.ndarray, list)):
                        serializable_metrics[key][k] = list(v)
                    elif isinstance(v, dict):
                        serializable_metrics[key][k] = {str(kk): vv for kk, vv in v.items()}
                    else:
                        serializable_metrics[key][k] = v
            else:
                serializable_metrics[key] = value

        # Save to file
        with open(os.path.join(self.results_dir, "study_results.json"), 'w') as f:
            json.dump({
                "study_name": self.study_name,
                "parameters": self.parameters,
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

    def run_full_study(self, num_episodes=10000, parameter_combinations=None):
        """Run a comprehensive study of Q-learning for Skud Pai Sho."""
        print(f"Starting comprehensive study: {self.study_name}")

        # 1. Train basic Q-learning agent
        print("\n1. Training basic Q-learning agent...")
        self.initialize_q_agent()
        self.train_q_agent(num_episodes=num_episodes, print_interval=num_episodes // 10)

        # 2. Perform state space analysis
        print("\n2. Performing state space analysis...")
        self.state_space_analysis(num_games=100)

        # 3. Perform action analysis
        print("\n3. Performing action analysis...")
        self.action_analysis(num_games=50)

        # 4. Parameter study (if parameter combinations provided)
        if parameter_combinations:
            print("\n4. Performing parameter study...")
            self.parameter_study(parameter_combinations, episodes_per_combo=num_episodes // 10)

        # 5. Compare with transformer (if available)
        try:
            print("\n5. Comparing with transformer model...")
            self.initialize_transformer()
            self.compare_q_vs_transformer(num_games=100)
        except:
            print("Skipping transformer comparison (missing dependencies).")

        # 6. Save all results
        print("\n6. Saving study results...")
        self.save_study_results()

        print(f"\nComprehensive study completed: {self.study_name}")
        return self.metrics