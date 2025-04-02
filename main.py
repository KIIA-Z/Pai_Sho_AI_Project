# main.py
import os
import torch
from game.state import SkudPaiShoState
from game.display import display_board, display_valid_moves, move_to_string
from ai_deprecated.utils import create_initial_model, get_ai_move
from ai_deprecated.training import train_skud_pai_sho_ai


def play_against_ai(model):
    """Interface for human vs AI gameplay."""
    state = SkudPaiShoState()
    human_player = 1  # By default, human is player 1

    print("Welcome to Skud Pai Sho!")
    print("You'll be playing against the AI.")
    player_choice = input("Do you want to play as Player 1 or Player 2? (1/2): ")
    if player_choice == "2":
        human_player = 2

    print(f"You are Player {human_player}.")
    print("Let's begin!\n")

    while not state.is_game_over() and state.turn_number < 200:  # Max 200 moves
        # Display current board
        display_board(state)
        print(f"\nTurn {state.turn_number + 1}: Player {state.current_player}'s move")

        if state.current_player == human_player:  # Human player's turn
            # Get valid moves
            valid_moves = state.get_valid_moves()

            if not valid_moves:
                print("No valid moves available. Game over.")
                break

            # Display valid moves
            display_valid_moves(valid_moves)

            # Get human move
            while True:
                try:
                    move_idx = int(input("\nEnter move number: "))
                    if 0 <= move_idx < len(valid_moves):
                        move = valid_moves[move_idx]
                        break
                    else:
                        print(f"Invalid move number. Please enter a number between 0 and {len(valid_moves) - 1}.")
                except ValueError:
                    print("Please enter a valid number.")

            print(f"You chose: {move_to_string(move)}")

        else:  # AI player's turn
            print("AI is thinking...")

            # Get AI move
            ai_move_result = get_ai_move(model, state)
            if ai_move_result is None:
                print("AI has no valid moves. Game over.")
                break

            move, value = ai_move_result

            # Display AI's evaluation
            evaluation = "favorable" if value > 0.3 else "neutral" if value > -0.3 else "unfavorable"
            print(f"AI evaluation: {evaluation} ({value:.2f})")
            print(f"AI plays: {move_to_string(move)}")

        # Make move
        state.make_move(move)

    # Game over
    display_board(state)
    print("\nGame Over!")

    winner = state.get_winner()
    if winner == 0:
        print("The game is a draw!")
    elif winner == human_player:
        print("Congratulations! You win!")
    else:
        print("The AI wins this time.")

    print(f"Final score - Player 1: {len(state.harmonies[1])} harmonies, Player 2: {len(state.harmonies[2])} harmonies")


def main():
    """Main function to run the Skud Pai Sho AI system."""
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    print("Skud Pai Sho AI")
    print("1. Train a new model")
    print("2. Play against a pre-trained model")
    print("3. Play against a randomly initialized model")
    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        # Train a new model
        print("\nTraining a new model...")
        iterations = int(input("Enter number of training iterations: "))
        games = int(input("Enter number of games per iteration: "))

        model = create_initial_model()
        model = train_skud_pai_sho_ai(model, iterations=iterations, games_per_iteration=games)

        print(f"Model trained for {iterations} iterations with {games} games each.")
        play_choice = input("Would you like to play against the trained model? (y/n): ")
        if play_choice.lower() == "y":
            play_against_ai(model)

    elif choice == "2":
        # Play against a pre-trained model
        model_files = [f for f in os.listdir("models") if f.endswith(".pth")]

        if not model_files:
            print("No pre-trained models found. You need to train a model first.")
            return

        print("\nAvailable models:")
        for i, model_file in enumerate(model_files):
            print(f"{i + 1}. {model_file}")

        model_idx = int(input(f"Select a model (1-{len(model_files)}): ")) - 1
        model_path = os.path.join("models", model_files[model_idx])

        try:
            model = create_initial_model()
            model.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
            play_against_ai(model)
        except Exception as e:
            print(f"Error loading model: {e}")

    elif choice == "3":
        # Play against a randomly initialized model
        print("\nPlaying against a randomly initialized model...")
        model = create_initial_model()
        play_against_ai(model)

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()