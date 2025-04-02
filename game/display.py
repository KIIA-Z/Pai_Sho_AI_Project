# game/display.py
from game.state import TileType, BOARD_SIZE


def display_board(state):
    """Display the current board state in ASCII art."""
    # Create symbols for different tile types
    symbols = {
        0: '·',  # Empty
        # Player 1 tiles (White Flower tiles)
        TileType.WHITE_LOTUS.value: 'L',
        TileType.LILY.value: 'I',
        TileType.JASMINE.value: 'J',
        TileType.RHODODENDRON.value: 'R',
        TileType.CHRYSANTHEMUM.value: 'C',
        TileType.WHITE_JADE.value: 'W',
        TileType.WHEEL.value: 'H',
        TileType.BOAT.value: 'B',
        TileType.ROCK.value: 'K',
        TileType.KNOTWEED.value: 'N',
        # Player 2 tiles (Accented Flower tiles)
        TileType.WHITE_LOTUS_ACCENT.value: 'l',
        TileType.LILY_ACCENT.value: 'i',
        TileType.JASMINE_ACCENT.value: 'j',
        TileType.RHODODENDRON_ACCENT.value: 'r',
        TileType.CHRYSANTHEMUM_ACCENT.value: 'c',
    }

    print("\n  Skud Pai Sho Board")
    print("  " + "-" * (BOARD_SIZE * 2 + 1))

    # Print column headers
    print("    ", end="")
    for x in range(BOARD_SIZE):
        print(f"{x % 10} ", end="")
    print()

    # Print rows
    for y in range(BOARD_SIZE):
        print(f"{y:2d} |", end="")
        for x in range(BOARD_SIZE):
            if state.is_on_board(x, y):
                tile = state.board[y, x]
                symbol = symbols.get(tile, '?')
                print(f"{symbol} ", end="")
            else:
                print("  ", end="")
        print("|")

    print("  " + "-" * (BOARD_SIZE * 2 + 1))

    # Print harmony count
    print(f"  Player 1 Harmonies: {len(state.harmonies[1])}")
    print(f"  Player 2 Harmonies: {len(state.harmonies[2])}")

    # Print available tiles
    print("\nAvailable Tiles:")
    for player in [1, 2]:
        print(f"Player {player}:", end=" ")
        for tile_type, count in state.available_tiles[player].items():
            if count > 0:
                symbol = symbols.get(tile_type.value, '?')
                print(f"{symbol}:{count}", end=" ")
        print()


def display_valid_moves(moves):
    """Display valid moves to the human player."""
    print("\nValid Moves:")
    for i, move in enumerate(moves):
        if move[0] == "plant":
            _, tile_type, x, y = move
            print(f"{i}: Plant {tile_type.name} at ({x}, {y})")
        elif move[0] == ("mo"
                         ""
                         "]"
                         ""
                         ""
                         "ve"):
            _, from_x, from_y, to_x, to_y = move
            print(f"{i}: Move from ({from_x}, {from_y}) to ({to_x}, {to_y})")


def move_to_string(move):
    """Convert a move to a human-readable string."""
    if move[0] == "plant":
        _, tile_type, x, y = move
        return f"Plant {tile_type.name} at ({x}, {y})"
    elif move[0] == "move":
        _, from_x, from_y, to_x, to_y = move
        return f"Move from ({from_x}, {from_y}) to ({to_x}, {to_y})"