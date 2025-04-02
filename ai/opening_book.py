# ai/opening_book.py
import json
import os
import random
import numpy as np
from collections import defaultdict
from game.state import TileType, BOARD_SIZE


class OpeningBook:
    """
    Opening book system for Skud Pai Sho.
    Stores and retrieves opening moves based on game state hashing.
    Can learn from self-play games and store successful openings.
    """

    def __init__(self, book_file=None):
        """
        Initialize opening book.

        Args:
            book_file: Optional file path to load an existing opening book
        """
        self.book = defaultdict(list)
        self.move_weights = defaultdict(dict)
        self.book_file = book_file or "data/opening_book.json"

        # Load existing book if available
        if os.path.exists(self.book_file):
            self.load()

    def hash_state(self, state):
        """
        Create a hash string for a game state.
        For opening book purposes, we only need the board positions.

        Args:
            state: Game state to hash

        Returns:
            String hash of the state
        """
        # For opening book, we only care about the board configuration
        # and whose turn it is
        board_str = str(state.board)
        player_str = str(state.current_player)
        turn_str = str(state.turn_number)

        # Combine into a hash
        return f"{board_str}_{player_str}_{turn_str}"

    def add_move(self, state, move, weight=1.0):
        """
        Add a move to the opening book.

        Args:
            state: Game state
            move: Move to add
            weight: Weight/score for this move (higher is better)
        """
        state_hash = self.hash_state(state)

        # Convert move to a serializable format
        move_str = str(move)

        # Update move weights
        if move_str in self.move_weights[state_hash]:
            # Exponential moving average of weights
            old_weight = self.move_weights[state_hash][move_str]
            self.move_weights[state_hash][move_str] = 0.9 * old_weight + 0.1 * weight
        else:
            self.move_weights[state_hash][move_str] = weight

        # Update the book (only keep moves with sufficient weight)
        self.book[state_hash] = [
            (move_str, w) for move_str, w in self.move_weights[state_hash].items()
            if w >= 0.4  # Threshold for keeping moves
        ]

    def get_move(self, state, temperature=0.5, threshold=0.0):
        """
        Get a move from the opening book for the current state.

        Args:
            state: Current game state
            temperature: Temperature for move selection (higher = more variety)
            threshold: Minimum weight threshold for considering moves

        Returns:
            A move if one is found in the book, otherwise None
        """
        state_hash = self.hash_state(state)

        if state_hash not in self.book or not self.book[state_hash]:
            return None

        # Get moves and weights
        moves_with_weights = self.book[state_hash]

        # Filter by threshold
        valid_moves = [(eval(move_str), weight) for move_str, weight in moves_with_weights
                       if weight >= threshold]

        if not valid_moves:
            return None

        if temperature == 0:
            # Deterministic selection (best move)
            return max(valid_moves, key=lambda x: x[1])[0]
        else:
            # Temperature-based selection
            moves, weights = zip(*valid_moves)
            weights = np.array(weights)

            # Apply temperature to weights
            weights = weights ** (1.0 / temperature)

            # Normalize to probabilities
            probs = weights / np.sum(weights)

            # Select move based on probabilities
            selected_idx = np.random.choice(len(moves), p=probs)
            return moves[selected_idx]

    def load(self):
        """Load opening book from file."""
        try:
            with open(self.book_file, 'r') as f:
                data = json.load(f)

                # Convert from saved format to internal format
                self.book = defaultdict(list)
                self.move_weights = defaultdict(dict)

                for state_hash, moves in data.items():
                    for move_str, weight in moves:
                        self.move_weights[state_hash][move_str] = weight

                    self.book[state_hash] = moves

                print(f"Loaded opening book with {len(self.book)} positions")
        except (FileNotFoundError, json.JSONDecodeError):
            print("No valid opening book found. Starting with empty book.")

    def save(self):
        """Save opening book to file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.book_file), exist_ok=True)

        # Save data
        with open(self.book_file, 'w') as f:
            json.dump(dict(self.book), f, indent=2)

        print(f"Saved opening book with {len(self.book)} positions")

    def learn_from_games(self, game_records, min_win_probability=0.55):
        """
        Learn openings from successful self-play games.

        Args:
            game_records: List of game records from self-play
            min_win_probability: Minimum win rate to include an opening
        """
        # Group games by opening sequence
        opening_results = defaultdict(list)

        for game in game_records:
            if not game:
                continue

            # Extract game result
            result = game[-1][2]  # Last state's reward

            # Extract first few moves (up to 10)
            opening_moves = []

            # Skip states that are already analyzed
            analyzed_hashes = set()

            for state, policy, value in game[:10]:  # First 10 positions
                state_hash = self.hash_state(state)

                if state_hash in analyzed_hashes:
                    continue

                analyzed_hashes.add(state_hash)

                # Get next move (if available)
                next_idx = game.index((state, policy, value)) + 1
                if next_idx < len(game):
                    next_state = game[next_idx][0]

                    # Determine what move was made
                    move = self._infer_move(state, next_state)
                    if move:
                        opening_moves.append((state_hash, move, state))

            # Record this opening sequence and its result
            opening_key = tuple((h, str(m)) for h, m, _ in opening_moves)
            opening_results[opening_key].append(result)

        # Add successful openings to the book
        openings_added = 0

        for opening, results in opening_results.items():
            # Calculate win probability
            win_count = sum(1 for r in results if r > 0)
            win_probability = win_count / len(results) if results else 0

            # Only add openings that win more than the threshold
            if win_probability >= min_win_probability:
                for state_hash, move_str, state in opening:
                    self.add_move(state, eval(move_str), weight=win_probability)
                    openings_added += 1

        print(f"Learned {openings_added} opening moves from {len(game_records)} games")

    def _infer_move(self, state1, state2):
        """
        Infer what move was made to go from state1 to state2.
        This is game-specific and needs to be implemented based on your game logic.

        For Skud Pai Sho, we need to detect either:
        1. A new piece was planted
        2. A piece was moved

        Args:
            state1: State before move
            state2: State after move

        Returns:
            The move that was made, or None if it can't be determined
        """
        # This is a simplified example - you'll need to adapt to your game logic
        # Implement based on how moves are represented in your game

        # For planting moves, there should be one more piece on the board
        # For moving moves, the total number of pieces remains the same

        # This is a placeholder implementation - you'll need to fill in the details
        # based on your specific game representation

        # Example:
        if hasattr(state1, 'last_move') and hasattr(state2, 'previous_move'):
            return state2.previous_move

        # Placeholder
        return None

    def get_stats(self):
        """Get statistics about the opening book."""
        total_positions = len(self.book)
        total_moves = sum(len(moves) for moves in self.book.values())
        avg_moves_per_position = total_moves / total_positions if total_positions > 0 else 0

        return {
            "total_positions": total_positions,
            "total_moves": total_moves,
            "avg_moves_per_position": avg_moves_per_position
        }


# Sample opening book entries for Skud Pai Sho
def create_sample_opening_book():
    """
    Create a sample opening book with common openings.
    This should be expanded with actual good openings discovered through play.
    """
    book = OpeningBook(book_file="data/sample_opening_book.json")

    # Create a starting state
    from game.state import SkudPaiShoState
    state = SkudPaiShoState()

    # Add some sample opening moves
    # First moves for player 1
    book.add_move(state, ("plant", TileType.FIRE, 4, 4), weight=0.9)
    book.add_move(state, ("plant", TileType.WATER, 3, 3), weight=0.8)
    book.add_move(state, ("plant", TileType.AIR, 4, 3), weight=0.7)

    # Save the book
    book.save()

    return book


# Function to integrate opening book with MCTS
# In opening_book.py, modify the get_move_with_opening_book_and_mcts function

def get_move_with_opening_book_and_mcts(model, state, opening_book, mcts_simulations=800, temperature=0.5):
    """
    Get a move using the opening book if available, otherwise fall back to MCTS.

    Args:
        model: Neural network model
        state: Current game state
        opening_book: Opening book object
        mcts_simulations: Number of MCTS simulations if opening book fails
        temperature: Temperature for move selection

    Returns:
        Selected move
    """
    # Check if we have a book move
    book_move = opening_book.get_move(state, temperature=temperature)

    if book_move:
        # We found a book move
        return book_move, None  # No policy needed for book moves

    # Fall back to MCTS
    from ai.mcts import mcts_search

    # Add debug info
    print(f"Using MCTS with model (input_channels={model.input_channels})")

    # Check state encoding shape
    encoded_state = state.encode_for_network()
    print(f"State encoding shape: {encoded_state.shape}")

    return mcts_search(model, state, num_simulations=mcts_simulations, temperature=temperature)