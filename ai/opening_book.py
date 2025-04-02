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

        # Check if state tracking already provides this information
        if hasattr(state2, 'last_move') and state2.last_move:
            return state2.last_move

        if hasattr(state2, 'previous_move') and state2.previous_move:
            return state2.previous_move

        # If we don't have direct move tracking, try to infer move from board difference
        try:
            # For planting: there should be exactly one more piece on the board
            # For moving: the total number of pieces stays the same but positions change

            # Compare board states to detect changes
            if hasattr(state1, 'board') and hasattr(state2, 'board'):
                board1 = state1.board
                board2 = state2.board

                # Find differences
                new_pieces = []
                moved_pieces = []
                removed_pieces = []

                for y in range(BOARD_SIZE):
                    for x in range(BOARD_SIZE):
                        if board1[y][x] == 0 and board2[y][x] != 0:
                            # A piece was added here
                            new_pieces.append((x, y, board2[y][x]))
                        elif board1[y][x] != 0 and board2[y][x] == 0:
                            # A piece was removed from here
                            removed_pieces.append((x, y, board1[y][x]))
                        elif board1[y][x] != 0 and board2[y][x] != 0 and board1[y][x] != board2[y][x]:
                            # A piece was changed here
                            moved_pieces.append((x, y, board1[y][x], board2[y][x]))

                # Try to reconstruct the move
                if len(new_pieces) == 1 and not removed_pieces:
                    # Planting move
                    x, y, tile_value = new_pieces[0]
                    # Convert value to TileType
                    tile_type = None
                    for t in TileType:
                        if t.value == tile_value:
                            tile_type = t
                            break

                    if tile_type:
                        return ("plant", tile_type, x, y)

                elif len(removed_pieces) == 1 and len(new_pieces) == 1:
                    # Moving move
                    from_x, from_y, _ = removed_pieces[0]
                    to_x, to_y, _ = new_pieces[0]
                    return ("move", from_x, from_y, to_x, to_y)

        except Exception as e:
            print(f"Error inferring move: {e}")

        # Couldn't determine the move
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


# Create a sample opening book with common openings
def create_sample_opening_book():
    """
    Create a sample opening book with common openings for Skud Pai Sho.
    Uses available tile types from the game implementation.
    """
    book = OpeningBook(book_file="data/sample_opening_book.json")

    # Create a starting state
    from game.state import SkudPaiShoState
    state = SkudPaiShoState()

    # Get the available tile types (excluding EMPTY which is usually 0)
    tile_types = list(TileType)
    if hasattr(TileType, 'EMPTY') and TileType.EMPTY in tile_types:
        tile_types.remove(TileType.EMPTY)
    elif 0 in [t.value for t in tile_types]:
        # Remove any tile type with value 0 (likely EMPTY)
        tile_types = [t for t in tile_types if t.value != 0]

    # If there are no tile types available, print warning and return
    if not tile_types:
        print("Warning: No tile types found in TileType enumeration.")
        return book

    # Add some sample opening moves for the center and nearby positions
    # First moves for player 1 (using whatever tile types are available)
    center = BOARD_SIZE // 2

    # Add moves for different tile types at strategic positions
    for i, tile_type in enumerate(tile_types[:3]):  # Use up to 3 different tile types
        # Strategic positions near center
        positions = [
            (center, center),  # Center
            (center - 1, center - 1),  # Top-left of center
            (center + 1, center - 1),  # Top-right of center
            (center - 1, center + 1),  # Bottom-left of center
            (center + 1, center + 1),  # Bottom-right of center
        ]

        if i < len(positions):
            x, y = positions[i]
            book.add_move(state, ("plant", tile_type, x, y), weight=0.9 - (i * 0.1))
            print(f"Added opening book move: plant {tile_type} at ({x},{y})")

    # Add a few more moves with lower weights
    if len(tile_types) > 0:
        book.add_move(state, ("plant", tile_types[0], 1, 1), weight=0.5)
        book.add_move(state, ("plant", tile_types[0], BOARD_SIZE - 2, BOARD_SIZE - 2), weight=0.5)

    # Save the book
    book.save()

    return book


# Function to integrate opening book with MCTS
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
        print(f"Using opening book move: {book_move}")
        return book_move, None  # No policy needed for book moves

    # Fall back to MCTS
    from ai.mcts import mcts_search

    # Add debug info
    print(f"Using MCTS with model (input_channels={model.input_channels})")

    # Check state encoding shape
    encoded_state = state.encode_for_network()
    print(f"State encoding shape: {encoded_state.shape}")

    return mcts_search(model, state, num_simulations=mcts_simulations, temperature=temperature)