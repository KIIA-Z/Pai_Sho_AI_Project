# game/state.py
import numpy as np
import math
from enum import Enum


# Define tile types as enums
class TileType(Enum):
    EMPTY = 0
    # White Lotus tiles (Harmony)
    WHITE_LOTUS = 1
    # Basic Flower tiles (Harmony)
    LILY = 2
    JASMINE = 3
    RHODODENDRON = 4
    CHRYSANTHEMUM = 5
    # Special tiles
    WHITE_JADE = 6
    WHEEL = 7
    BOAT = 8
    ROCK = 9
    KNOTWEED = 10
    # Accented tiles for player distinction
    WHITE_LOTUS_ACCENT = 11
    LILY_ACCENT = 12
    JASMINE_ACCENT = 13
    RHODODENDRON_ACCENT = 14
    CHRYSANTHEMUM_ACCENT = 15


# Game constants
BOARD_SIZE = 17  # 17x17 circular board
BOARD_RADIUS = 8  # 8 spaces from center in any direction
MAX_MOVES = 200  # Maximum number of moves in a typical game


class SkudPaiShoState:
    def __init__(self):
        # Initialize the board (0 = empty, positive = player 1 tiles, negative = player 2 tiles)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1  # 1 or 2
        self.turn_number = 0
        self.history = []  # List of previous states

        # Available tiles for each player (using standard Skud Pai Sho counts)
        self.available_tiles = {
            1: {
                TileType.WHITE_LOTUS: 1,
                TileType.LILY: 3,
                TileType.JASMINE: 3,
                TileType.RHODODENDRON: 2,
                TileType.CHRYSANTHEMUM: 2,
                TileType.WHITE_JADE: 3,
                TileType.WHEEL: 3,
                TileType.BOAT: 2,
                TileType.ROCK: 1,
                TileType.KNOTWEED: 1
            },
            2: {
                TileType.WHITE_LOTUS_ACCENT: 1,
                TileType.LILY_ACCENT: 3,
                TileType.JASMINE_ACCENT: 3,
                TileType.RHODODENDRON_ACCENT: 2,
                TileType.CHRYSANTHEMUM_ACCENT: 2,
                TileType.WHITE_JADE: 3,
                TileType.WHEEL: 3,
                TileType.BOAT: 2,
                TileType.ROCK: 1,
                TileType.KNOTWEED: 1
            }
        }

        # Initialize the center point (gate)
        self.center = (BOARD_SIZE // 2, BOARD_SIZE // 2)

        # Track harmonies for scoring
        self.harmonies = {1: [], 2: []}

        # Game state
        self.game_over = False
        self.winner = None

    def is_on_board(self, x, y):
        """Check if a point is on the circular board."""
        # Distance from center
        dx = x - self.center[0]
        dy = y - self.center[1]
        distance = math.sqrt(dx * dx + dy * dy)

        return distance <= BOARD_RADIUS

    def is_harmony_tile(self, tile_type):
        """Check if a tile can create harmonies."""
        harmony_tiles = [
            TileType.WHITE_LOTUS, TileType.LILY, TileType.JASMINE,
            TileType.RHODODENDRON, TileType.CHRYSANTHEMUM,
            TileType.WHITE_LOTUS_ACCENT, TileType.LILY_ACCENT, TileType.JASMINE_ACCENT,
            TileType.RHODODENDRON_ACCENT, TileType.CHRYSANTHEMUM_ACCENT
        ]
        return tile_type in harmony_tiles

    def get_tile_owner(self, x, y):
        """Return the owner of a tile (1, 2, or 0 if empty)."""
        tile = self.board[y, x]
        if tile == 0:
            return 0
        elif 1 <= tile <= 10:  # Player 1 tiles
            return 1
        else:  # Player 2 tiles (11-20)
            return 2

    def get_tile_type(self, x, y):
        """Return the type of tile at position (x, y)."""
        return self.board[y, x]

    def get_movement_range(self, tile_type):
        """Return the movement range for a specific tile type."""
        # Define movement ranges according to Skud Pai Sho rules
        if tile_type in [TileType.WHITE_LOTUS, TileType.WHITE_LOTUS_ACCENT]:
            return 1
        elif tile_type in [TileType.LILY, TileType.LILY_ACCENT]:
            return 1
        elif tile_type in [TileType.JASMINE, TileType.JASMINE_ACCENT]:
            return 2
        elif tile_type in [TileType.RHODODENDRON, TileType.RHODODENDRON_ACCENT]:
            return 3
        elif tile_type in [TileType.CHRYSANTHEMUM, TileType.CHRYSANTHEMUM_ACCENT]:
            return 4
        elif tile_type == TileType.WHITE_JADE:
            return 2
        elif tile_type == TileType.WHEEL:
            return 2
        elif tile_type == TileType.BOAT:
            return 6
        elif tile_type == TileType.ROCK:
            return 0  # Rocks can't move
        elif tile_type == TileType.KNOTWEED:
            return 1
        else:
            return 0

    def get_valid_moves(self):
        """Return a list of valid moves in the current state."""
        moves = []

        # First 2 turns are planting turns (placing on your side of the board)
        if self.turn_number < 2:
            # Get valid planting positions
            for y in range(BOARD_SIZE):
                for x in range(BOARD_SIZE):
                    # Check if position is on the board
                    if not self.is_on_board(x, y):
                        continue

                    # Check if position is empty
                    if self.board[y, x] != 0:
                        continue

                    # Check if position is on the player's side (based on gates)
                    on_player_side = False
                    if self.current_player == 1 and y < self.center[1]:
                        on_player_side = True
                    elif self.current_player == 2 and y > self.center[1]:
                        on_player_side = True

                    if on_player_side:
                        # Add planting moves for each available tile
                        for tile_type, count in self.available_tiles[self.current_player].items():
                            if count > 0:
                                moves.append(("plant", tile_type, x, y))
        else:
            # Regular turns: can either plant a tile or move an existing one

            # Get planting moves
            for y in range(BOARD_SIZE):
                for x in range(BOARD_SIZE):
                    # Check if position is on the board, next to a gate, and empty
                    if (self.is_on_board(x, y) and self.board[y, x] == 0 and
                            (abs(x - self.center[0]) + abs(y - self.center[1]) == 1)):
                        # Add planting moves for each available tile
                        for tile_type, count in self.available_tiles[self.current_player].items():
                            if count > 0:
                                moves.append(("plant", tile_type, x, y))

            # Get movement moves
            for y in range(BOARD_SIZE):
                for x in range(BOARD_SIZE):
                    # Check if position has the current player's tile
                    if self.get_tile_owner(x, y) == self.current_player:
                        tile_type = self.get_tile_type(x, y)
                        movement_range = self.get_movement_range(TileType(tile_type))

                        # If the tile can move, add movement options
                        if movement_range > 0:
                            # Check each direction (orthogonal and diagonal)
                            directions = [
                                (1, 0), (0, 1), (-1, 0), (0, -1),  # Orthogonal
                                (1, 1), (-1, 1), (1, -1), (-1, -1)  # Diagonal
                            ]

                            for dx, dy in directions:
                                for dist in range(1, movement_range + 1):
                                    new_x, new_y = x + dx * dist, y + dy * dist

                                    # Check if the new position is valid
                                    if (0 <= new_x < BOARD_SIZE and 0 <= new_y < BOARD_SIZE and
                                            self.is_on_board(new_x, new_y) and self.board[new_y, new_x] == 0):
                                        # Check if the path is clear
                                        path_clear = True
                                        for i in range(1, dist):
                                            path_x, path_y = x + dx * i, y + dy * i
                                            if self.board[path_y, path_x] != 0:
                                                path_clear = False
                                                break

                                        if path_clear:
                                            moves.append(("move", x, y, new_x, new_y))
                                    else:
                                        # Stop checking this direction if blocked
                                        break

        return moves

    def make_move(self, move):
        """Apply a move to the current state and return a new state."""
        # Store current state in history
        self.history.append(self.copy())

        move_type = move[0]

        if move_type == "plant":
            _, tile_type, x, y = move
            # Place the tile and decrement available count
            self.board[y, x] = tile_type.value
            self.available_tiles[self.current_player][tile_type] -= 1

        elif move_type == "move":
            _, from_x, from_y, to_x, to_y = move
            # Move the tile
            tile_type = self.board[from_y, from_x]
            self.board[to_y, to_x] = tile_type
            self.board[from_y, from_x] = 0

        # Update harmonies
        self.update_harmonies()

        # Check if game is over
        self.check_game_over()

        # Switch player
        self.current_player = 3 - self.current_player  # Toggles between 1 and 2
        self.turn_number += 1

        return self

    def count_harmonies(self):
        """
        Count the total number of harmonies on the board.
        Returns the sum of harmonies for both players.
        """
        # Simply return the total number of harmonies already tracked in the state
        return len(self.harmonies[1]) + len(self.harmonies[2])

    def update_harmonies(self):
        """Update the harmonies on the board."""
        # Clear current harmonies
        self.harmonies = {1: [], 2: []}

        # Check for harmonies in both orthogonal and diagonal directions
        directions = [
            (1, 0), (0, 1),  # Orthogonal (horizontal, vertical)
            (1, 1), (1, -1)  # Diagonal
        ]

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                # Skip empty spaces
                if self.board[y, x] == 0:
                    continue

                # Get owner and check if it's a harmony tile
                owner = self.get_tile_owner(x, y)
                tile_type = self.get_tile_type(x, y)

                if not self.is_harmony_tile(TileType(tile_type)):
                    continue

                # Check each direction for harmonies
                for dx, dy in directions:
                    # Need to find another harmony tile of the same player
                    harmony_found = False
                    harmony_positions = []

                    # Check up to 4 spaces in this direction (max distance between harmony tiles)
                    for dist in range(1, 5):
                        new_x, new_y = x + dx * dist, y + dy * dist

                        # Check boundaries
                        if not (0 <= new_x < BOARD_SIZE and 0 <= new_y < BOARD_SIZE and self.is_on_board(new_x, new_y)):
                            break

                        new_owner = self.get_tile_owner(new_x, new_y)
                        new_tile_type = self.get_tile_type(new_x, new_y)

                        # If we hit an opponent's tile or a non-harmony tile, harmony is broken
                        if new_owner != 0 and (new_owner != owner or not self.is_harmony_tile(TileType(new_tile_type))):
                            break

                        # If we find another of our harmony tiles, we have a harmony
                        if new_owner == owner and self.is_harmony_tile(TileType(new_tile_type)):
                            harmony_found = True
                            harmony_positions = [(x, y), (new_x, new_y)]
                            break

                    # If we found a harmony, add it to the list
                    if harmony_found:
                        # Check if this harmony is already counted
                        new_harmony = True
                        for existing_harmony in self.harmonies[owner]:
                            if set(harmony_positions) == set(existing_harmony):
                                new_harmony = False
                                break

                        if new_harmony:
                            self.harmonies[owner].append(harmony_positions)

    def check_game_over(self):
        """Check if the game is over based on Skud Pai Sho rules."""
        # Victory condition: 4 or more harmonies
        for player in [1, 2]:
            if len(self.harmonies[player]) >= 4:
                self.game_over = True
                self.winner = player
                return

        # Check if maximum moves reached
        if self.turn_number >= MAX_MOVES:
            self.game_over = True
            # Determine winner based on harmony count
            if len(self.harmonies[1]) > len(self.harmonies[2]):
                self.winner = 1
            elif len(self.harmonies[2]) > len(self.harmonies[1]):
                self.winner = 2
            else:
                self.winner = 0  # Draw

    def is_game_over(self):
        """Return whether the game is over."""
        return self.game_over

    def get_winner(self):
        """Return the winner of the game (0 for draw)."""
        return self.winner

    def get_reward(self, player):
        """Get the reward for a player (-1 to 1)."""
        if not self.game_over:
            return 0

        if self.winner == 0:  # Draw
            return 0

        return 1 if self.winner == player else -1

    def copy(self):
        """Create a deep copy of the current state."""
        new_state = SkudPaiShoState()
        new_state.board = self.board.copy()
        new_state.current_player = self.current_player
        new_state.turn_number = self.turn_number
        new_state.history = self.history.copy()

        # Deep copy of available tiles
        new_state.available_tiles = {
            1: self.available_tiles[1].copy(),
            2: self.available_tiles[2].copy()
        }

        # Copy harmonies
        new_state.harmonies = {
            1: [h.copy() for h in self.harmonies[1]],
            2: [h.copy() for h in self.harmonies[2]]
        }

        new_state.game_over = self.game_over
        new_state.winner = self.winner

        return new_state


    def detect_cycles(self, last_n=8):
        """
        Check if the current state has been repeated recently.

        Args:
            last_n: Number of past states to check

        Returns:
            bool: True if a cycle was detected, False otherwise
        """
        if len(self.history) < 2:
            return False

        # Create a hash of the current board state
        current_state_hash = hash(self.board.tobytes())

        # Check against recent history (limiting to last_n states)
        history_to_check = self.history[-last_n:] if len(self.history) > last_n else self.history

        for past_state in history_to_check:
            if hash(past_state.board.tobytes()) == current_state_hash:
                return True

        return False

    def encode_for_network(self):
        """Encode the game state as input for the neural network."""
        # Create channels for each tile type, player turn, and game state info
        channels = []

        # Encode each tile type for each player (21 channels)
        for tile_type in TileType:
            if tile_type == TileType.EMPTY:
                continue

            # Player 1 tiles of this type
            channels.append((self.board == tile_type.value).astype(np.float32))

        # Encode turn number (normalized)
        turn_channel = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32) * (self.turn_number / MAX_MOVES)
        channels.append(turn_channel)

        # Encode current player
        player_channel = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32) * (
                    self.current_player - 1.5) * 2  # Maps 1->-1, 2->1
        channels.append(player_channel)

        # Encode harmony count for both players
        for player in [1, 2]:
            harmony_channel = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32) * (
                        len(self.harmonies[player]) / 4.0)  # Normalize by win condition
            channels.append(harmony_channel)

        # Encode the board region mask (1 for valid positions, 0 for off-board)
        board_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.is_on_board(x, y):
                    board_mask[y, x] = 1.0
        channels.append(board_mask)

        # Stack all channels
        return np.stack(channels)