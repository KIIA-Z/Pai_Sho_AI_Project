# q_learning.py
import numpy as np
import pickle
import os
import sys
import random
from collections import defaultdict

# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add it to the path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from game.state import SkudPaiShoState


class SkudPaiShoQLearning:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0,
                 exploration_decay=0.995, min_exploration_rate=0.1):
        """
        Initialize the Q-learning agent.

        Args:
            learning_rate: Alpha - how much to update Q-values based on new information
            discount_factor: Gamma - how much to value future rewards
            exploration_rate: Epsilon - probability of taking a random action
            exploration_decay: How quickly to reduce exploration
            min_exploration_rate: Minimum exploration rate
        """
        self.q_table = defaultdict(lambda: defaultdict(float))  # Q(s, a) table
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate

        # Statistics for evaluation
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.games_played = 0
        self.avg_game_length = 0
        self.avg_harmonies = 0

    def get_state_key(self, state):
        """
        Convert the game state to a hashable key for the Q-table.
        This is challenging for Skud Pai Sho due to the large state space.
        We'll use a simplified representation focusing on essential elements.
        """
        # Get positions of all pieces (simplified)
        piece_positions = []
        for y in range(state.board.shape[0]):
            for x in range(state.board.shape[1]):
                if state.board[y, x] != 0:
                    piece_positions.append((x, y, state.board[y, x]))

        # Current player
        current_player = state.current_player

        # Harmony counts
        harmonies_p1 = len(state.harmonies[1])
        harmonies_p2 = len(state.harmonies[2])

        # Turn number (binned to reduce state space)
        turn_bin = min(state.turn_number // 5, 10)  # Bin into groups of 5 turns, max 10 bins

        # Create a tuple that represents the essential game state
        state_key = (tuple(sorted(piece_positions)), current_player, harmonies_p1, harmonies_p2, turn_bin)
        return state_key

    def get_action_key(self, action):
        """Convert an action to a hashable key for the Q-table."""
        return str(action)  # Simple string representation

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current game state

        Returns:
            Chosen action
        """
        valid_moves = state.get_valid_moves()

        if not valid_moves:
            return None

        # With probability epsilon, choose a random action (explore)
        if random.random() < self.exploration_rate:
            return random.choice(valid_moves)

        # Otherwise, choose the best action based on Q-values (exploit)
        state_key = self.get_state_key(state)

        # Get Q-values for all valid actions
        q_values = [self.q_table[state_key][self.get_action_key(move)] for move in valid_moves]

        # If all Q-values are the same (e.g., all 0), choose randomly
        if len(set(q_values)) == 1:
            return random.choice(valid_moves)

        # Otherwise, choose action with highest Q-value
        best_index = np.argmax(q_values)
        return valid_moves[best_index]

    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value for a state-action pair using the Q-learning formula.

        Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        state_key = self.get_state_key(state)
        action_key = self.get_action_key(action)

        # Get current Q-value
        current_q = self.q_table[state_key][action_key]

        # Get max Q-value for next state
        next_state_key = self.get_state_key(next_state)

        next_valid_moves = next_state.get_valid_moves()
        if next_valid_moves:
            # If there are valid moves, get max Q-value
            next_q_values = [self.q_table[next_state_key][self.get_action_key(move)]
                             for move in next_valid_moves]
            max_next_q = max(next_q_values)
        else:
            # If no valid moves (terminal state), max Q-value is 0
            max_next_q = 0

        # Calculate new Q-value
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

        # Update Q-table
        self.q_table[state_key][action_key] = new_q

    def decay_exploration(self):
        """Decay exploration rate."""
        self.exploration_rate = max(self.min_exploration_rate,
                                    self.exploration_rate * self.exploration_decay)

    def calculate_reward(self, state, player):
        """
        Calculate reward for the current state.
        We'll use a reward function that encourages creating harmonies.

        Args:
            state: Current game state
            player: Player for which to calculate reward

        Returns:
            Reward value
        """
        # Base reward
        reward = 0

        # Reward for creating harmonies
        reward += len(state.harmonies[player]) * 0.5

        # Penalty for opponent harmonies
        opponent = 3 - player  # 1 -> 2, 2 -> 1
        reward -= len(state.harmonies[opponent]) * 0.3

        # Reward for winning
        if state.is_game_over():
            if state.get_winner() == player:
                reward += 10  # Big reward for winning
            elif state.get_winner() == opponent:
                reward -= 10  # Big penalty for losing
            elif state.get_winner() == 0:
                reward += 1  # Small reward for a draw

        return reward

    def train_episode(self):
        """
        Train the agent for one episode (game).

        Returns:
            Tuple of (winner, game_length, final_harmonies)
        """
        state = SkudPaiShoState()
        game_length = 0

        while not state.is_game_over() and game_length < 200:  # Max 200 moves
            # Current player
            current_player = state.current_player

            # Choose action
            action = self.choose_action(state)

            if action is None:
                break  # No valid moves

            # Apply action
            old_state = state.copy()
            state.make_move(action)
            game_length += 1

            # Calculate reward
            reward = self.calculate_reward(state, current_player)

            # Update Q-value
            self.update_q_value(old_state, action, reward, state)

        # Update statistics
        winner = state.get_winner()
        self.games_played += 1

        if winner == 1:
            self.wins += 1
        elif winner == 2:
            self.losses += 1
        else:
            self.draws += 1

        # Update average game length
        self.avg_game_length = ((self.avg_game_length * (self.games_played - 1)) + game_length) / self.games_played

        # Update average harmonies
        total_harmonies = len(state.harmonies[1]) + len(state.harmonies[2])
        self.avg_harmonies = ((self.avg_harmonies * (self.games_played - 1)) + total_harmonies) / self.games_played

        # Decay exploration rate
        self.decay_exploration()

        return winner, game_length, (len(state.harmonies[1]), len(state.harmonies[2]))

    def train(self, num_episodes=1000, print_interval=100):
        """
        Train the agent for multiple episodes.

        Args:
            num_episodes: Number of episodes (games) to train
            print_interval: How often to print statistics
        """
        print(f"Starting Q-learning training for {num_episodes} episodes...")

        for episode in range(1, num_episodes + 1):
            winner, game_length, harmonies = self.train_episode()

            if episode % print_interval == 0:
                win_rate = self.wins / self.games_played
                loss_rate = self.losses / self.games_played
                draw_rate = self.draws / self.games_played

                print(
                    f"Episode {episode}/{num_episodes}, Win rate: {win_rate:.3f}, Loss rate: {loss_rate:.3f}, Draw rate: {draw_rate:.3f}")
                print(
                    f"Exploration rate: {self.exploration_rate:.3f}, Avg game length: {self.avg_game_length:.1f}, Avg harmonies: {self.avg_harmonies:.1f}")
                print(f"Q-table entries: {len(self.q_table)}")
                print("-" * 50)

        print("Training completed!")
        print(
            f"Final statistics - Win rate: {self.wins / self.games_played:.3f}, Loss rate: {self.losses / self.games_played:.3f}, Draw rate: {self.draws / self.games_played:.3f}")
        print(f"Average game length: {self.avg_game_length:.1f}, Average harmonies: {self.avg_harmonies:.1f}")
        print(f"Total Q-table entries: {len(self.q_table)}")

    def save(self, filename="q_learning_model.pkl"):
        """Save the Q-table and statistics to a file."""
        data = {
            'q_table': dict(self.q_table),  # Convert defaultdict to regular dict
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate,
            'exploration_decay': self.exploration_decay,
            'min_exploration_rate': self.min_exploration_rate,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'games_played': self.games_played,
            'avg_game_length': self.avg_game_length,
            'avg_harmonies': self.avg_harmonies
        }

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        print(f"Model saved to {filename}")

    def load(self, filename="q_learning_model.pkl"):
        """Load the Q-table and statistics from a file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Convert regular dict to defaultdict
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_key, actions in data['q_table'].items():
            for action_key, q_value in actions.items():
                self.q_table[state_key][action_key] = q_value

        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.exploration_rate = data['exploration_rate']
        self.exploration_decay = data['exploration_decay']
        self.min_exploration_rate = data['min_exploration_rate']
        self.wins = data['wins']
        self.losses = data['losses']
        self.draws = data['draws']
        self.games_played = data['games_played']
        self.avg_game_length = data['avg_game_length']
        self.avg_harmonies = data['avg_harmonies']

        print(f"Model loaded from {filename}")
        print(f"Loaded Q-table with {len(self.q_table)} entries")

    def get_action_distribution(self, state):
        """
        Get the distribution of Q-values for all valid actions in a state.
        Useful for analysis.

        Args:
            state: Game state

        Returns:
            Dictionary mapping actions to Q-values
        """
        state_key = self.get_state_key(state)
        valid_moves = state.get_valid_moves()

        distribution = {}
        for move in valid_moves:
            action_key = self.get_action_key(move)
            distribution[action_key] = self.q_table[state_key][action_key]

        return distribution

    def play_against_self(self, num_games=100):
        """
        Make the agent play against itself to evaluate performance.

        Args:
            num_games: Number of games to play

        Returns:
            Dictionary with statistics
        """
        print(f"Playing {num_games} self-play games for evaluation...")

        # Set exploration rate to minimum for evaluation
        old_exploration_rate = self.exploration_rate
        self.exploration_rate = self.min_exploration_rate

        wins_p1 = 0
        wins_p2 = 0
        draws = 0
        game_lengths = []
        harmony_counts = []

        for game in range(num_games):
            state = SkudPaiShoState()
            game_length = 0

            while not state.is_game_over() and game_length < 200:
                action = self.choose_action(state)

                if action is None:
                    break

                state.make_move(action)
                game_length += 1

            # Record statistics
            winner = state.get_winner()
            if winner == 1:
                wins_p1 += 1
            elif winner == 2:
                wins_p2 += 1
            else:
                draws += 1

            game_lengths.append(game_length)
            harmony_counts.append(len(state.harmonies[1]) + len(state.harmonies[2]))

            if (game + 1) % 10 == 0:
                print(f"Evaluated {game + 1}/{num_games} games...")

        # Restore exploration rate
        self.exploration_rate = old_exploration_rate

        # Compile statistics
        stats = {
            'wins_p1': wins_p1,
            'wins_p2': wins_p2,
            'draws': draws,
            'win_rate_p1': wins_p1 / num_games,
            'win_rate_p2': wins_p2 / num_games,
            'draw_rate': draws / num_games,
            'avg_game_length': sum(game_lengths) / len(game_lengths),
            'avg_harmonies': sum(harmony_counts) / len(harmony_counts)
        }

        print("Evaluation completed!")
        print(f"Player 1 win rate: {stats['win_rate_p1']:.3f}")
        print(f"Player 2 win rate: {stats['win_rate_p2']:.3f}")
        print(f"Draw rate: {stats['draw_rate']:.3f}")
        print(f"Average game length: {stats['avg_game_length']:.1f}")
        print(f"Average harmonies: {stats['avg_harmonies']:.1f}")

        return stats