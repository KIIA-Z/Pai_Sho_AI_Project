# main.py
import os
import torch
import random
from game.state import SkudPaiShoState, TileType, BOARD_SIZE
from game.display import display_board, display_valid_moves, move_to_string
from ai.model import SkudPaiShoTransformer
from ai.utils import get_ai_move, create_initial_model
from ai.opening_book import OpeningBook
from ai.integrated_training import train_skud_pai_sho_ai_with_mcts


def play_against_ai(model, use_mcts=True, mcts_simulations=400, use_opening_book=False):
    """Interface for human vs AI gameplay with MCTS and opening book support."""
    state = SkudPaiShoState()
    human_player = 1  # By default, human is player 1

    # Initialize opening book if requested
    opening_book = None
    if use_opening_book:
        opening_book_path = "data/opening_book.json"
        if os.path.exists(opening_book_path):
            opening_book = OpeningBook(book_file=opening_book_path)
            print("Using opening book for AI moves.")
        else:
            print("Opening book file not found. Playing without opening book.")

    print("Welcome to Skud Pai Sho!")
    print("You'll be playing against the AI.")
    player_choice = input("Do you want to play as Player 1 or Player 2? (1/2): ")
    if player_choice == "2":
        human_player = 2

    print(f"You are Player {human_player}.")
    if use_mcts:
        print(f"AI is using MCTS with {mcts_simulations} simulations per move.")
    print("Let's begin!\n")

    # Store game history for potential analysis
    game_history = []

    while not state.is_game_over() and state.turn_number < 200:  # Max 200 moves
        # Add current state to history
        game_history.append(state.copy())

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

            # Get AI move using new method with MCTS and opening book
            move, value = get_ai_move(
                model,
                state,
                temperature=0.0,
                deterministic=True,
                mcts_simulations=mcts_simulations if use_mcts else 0,
                opening_book=opening_book
            )

            if move is None:
                print("AI has no valid moves. Game over.")
                break

            # Display AI's evaluation
            evaluation = "favorable" if value > 0.3 else "neutral" if value > -0.3 else "unfavorable"
            print(f"AI evaluation: {evaluation} ({value:.2f})")
            print(f"AI plays: {move_to_string(move)}")

        # Make move
        state.make_move(move)

    # Add final state to history
    game_history.append(state.copy())

    # Game over
    display_board(state)
    print("\nGame Over!")

    winner = state.get_winner() if hasattr(state, 'get_winner') else (
        1 if state.get_reward(1) > 0 else 2 if state.get_reward(1) < 0 else 0
    )

    if winner == 0:
        print("The game is a draw!")
    elif winner == human_player:
        print("Congratulations! You win!")
    else:
        print("The AI wins this time.")

    harmony_count1 = len(state.harmonies[1]) if hasattr(state, 'harmonies') else state.count_harmonies(1)
    harmony_count2 = len(state.harmonies[2]) if hasattr(state, 'harmonies') else state.count_harmonies(2)
    print(f"Final score - Player 1: {harmony_count1} harmonies, Player 2: {harmony_count2} harmonies")

    # Option to save the game
    save_option = input("Would you like to save this game for AI learning? (y/n): ")
    if save_option.lower() == 'y':
        game_dir = "game_records"
        os.makedirs(game_dir, exist_ok=True)
        timestamp = torch.datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(game_history, os.path.join(game_dir, f"game_{timestamp}.pt"))
        print(f"Game saved to game_records/game_{timestamp}.pt")


def main():
    """Main function to run the Skud Pai Sho AI system."""
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    print("Skud Pai Sho AI")
    print("1. Train a new model with MCTS")
    print("2. Play against a pre-trained model")
    print("3. Play against a randomly initialized model")
    print("4. Advanced options")
    choice = input("Enter your choice (1/2/3/4): ")

    if choice == "1":
        # Train a new model using MCTS
        print("\nTraining a new model with MCTS...")

        # Get training parameters
        iterations = int(input("Enter number of training iterations: "))
        games = int(input("Enter number of games per iteration: "))
        mcts_sims = int(input("Enter number of MCTS simulations per move (recommended: 400-1200): "))
        use_opening_book = input("Use opening book? (y/n): ").lower() == 'y'

        print("\nInitializing model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Calculate input channels based on game requirements
        input_channels = 20  # Adjust this based on your game state encoding
        model = create_initial_model(input_channels=input_channels, device=device)

        # Train model using the integrated training function with MCTS
        model, training_log = train_skud_pai_sho_ai_with_mcts(
            model,
            iterations=iterations,
            games_per_iteration=games,
            mcts_simulations=mcts_sims,
            use_opening_book=use_opening_book
        )

        print(f"Model trained for {iterations} iterations with {games} games each.")
        play_choice = input("Would you like to play against the trained model? (y/n): ")
        if play_choice.lower() == "y":
            use_mcts_play = input("Use MCTS for AI gameplay? (y/n): ").lower() == 'y'
            mcts_sims_play = int(
                input("Enter MCTS simulations for gameplay (recommended: 200-400): ")) if use_mcts_play else 0
            play_against_ai(model, use_mcts=use_mcts_play, mcts_simulations=mcts_sims_play,
                            use_opening_book=use_opening_book)

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
        if model_idx < 0 or model_idx >= len(model_files):
            print("Invalid selection.")
            return

        model_path = os.path.join("models", model_files[model_idx])

        # MCTS options
        use_mcts = input("Use MCTS for stronger play? (y/n): ").lower() == 'y'
        mcts_sims = int(input("Enter number of MCTS simulations (100-1000, higher = stronger): ")) if use_mcts else 0
        use_opening_book = input("Use opening book if available? (y/n): ").lower() == 'y'

        try:
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

            # Determine if this is a state dict or full checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Try to get additional info if available
                iteration = checkpoint.get('iteration', 'unknown')
                win_rate = checkpoint.get('win_rate', 'unknown')
                print(f"Model from iteration {iteration}, win rate: {win_rate}")
            else:
                state_dict = checkpoint

            # Create model with correct parameters
            input_channels = 20  # You may need to adjust this
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Try to infer model parameters from the state dict
            first_conv_weight = state_dict.get('conv1.weight')
            if first_conv_weight is not None:
                input_channels = first_conv_weight.shape[1]

            # Create model
            model = create_initial_model(input_channels=input_channels, device=device)
            model.load_state_dict(state_dict)
            model.eval()  # Set to evaluation mode

            print(f"Model loaded from {model_path}")
            play_against_ai(model, use_mcts=use_mcts, mcts_simulations=mcts_sims, use_opening_book=use_opening_book)
        except Exception as e:
            print(f"Error loading model: {e}")

    elif choice == "3":
        # Play against a randomly initialized model
        print("\nPlaying against a randomly initialized model...")

        # MCTS options
        use_mcts = input("Use MCTS? (y/n): ").lower() == 'y'
        mcts_sims = int(input("Enter number of MCTS simulations (recommended: 200-400): ")) if use_mcts else 0

        # Create model with random weights
        input_channels = 20  # You may need to adjust this
        model = create_initial_model(input_channels=input_channels)

        play_against_ai(model, use_mcts=use_mcts, mcts_simulations=mcts_sims, use_opening_book=False)

    elif choice == "4":
        # Advanced options
        print("\nAdvanced Options:")
        print("1. Create initial opening book")
        print("2. Evaluate model strength")
        print("3. Analyze saved games")
        print("4. Back to main menu")

        advanced_choice = input("Enter your choice (1-4): ")

        if advanced_choice == "1":
            # Create initial opening book
            from ai.opening_book import create_sample_opening_book
            print("Creating sample opening book...")
            opening_book = create_sample_opening_book()
            print(f"Sample opening book created with {opening_book.get_stats()['total_positions']} positions.")

        elif advanced_choice == "2":
            # Evaluate model strength
            model_files = [f for f in os.listdir("models") if f.endswith(".pth")]
            if not model_files:
                print("No models found for evaluation.")
                return

            print("\nAvailable models:")
            for i, model_file in enumerate(model_files):
                print(f"{i + 1}. {model_file}")

            model_idx = int(input(f"Select a model to evaluate (1-{len(model_files)}): ")) - 1
            if model_idx < 0 or model_idx >= len(model_files):
                print("Invalid selection.")
                return

            model_path = os.path.join("models", model_files[model_idx])

            num_games = int(input("Number of games for evaluation (e.g., 50): "))
            mcts_sims = int(input("MCTS simulations per move (0 for no MCTS): "))

            try:
                # Load the model
                checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                input_channels = 20
                first_conv_weight = state_dict.get('conv1.weight')
                if first_conv_weight is not None:
                    input_channels = first_conv_weight.shape[1]

                model = create_initial_model(input_channels=input_channels)
                model.load_state_dict(state_dict)
                model.eval()

                from ai.utils import evaluate_model
                print(f"Evaluating model over {num_games} games...")
                results = evaluate_model(
                    model,
                    num_games=num_games,
                    opponent="random",
                    mcts_simulations=mcts_sims
                )

                print("\nEvaluation Results:")
                print(f"Win rate: {results['win_rate']:.3f}")
                print(f"Loss rate: {results['loss_rate']:.3f}")
                print(f"Draw rate: {results['draw_rate']:.3f}")
                print(f"Average game length: {results['avg_game_length']:.1f} moves")

            except Exception as e:
                print(f"Error during evaluation: {e}")

        elif advanced_choice == "3":
            # Analyze saved games
            game_dir = "game_records"
            if not os.path.exists(game_dir):
                print("No saved games found.")
                return

            game_files = [f for f in os.listdir(game_dir) if f.endswith(".pt")]
            if not game_files:
                print("No saved games found.")
                return

            print("\nAvailable saved games:")
            for i, game_file in enumerate(game_files):
                print(f"{i + 1}. {game_file}")

            game_idx = int(input(f"Select a game to analyze (1-{len(game_files)}): ")) - 1
            if game_idx < 0 or game_idx >= len(game_files):
                print("Invalid selection.")
                return

            game_path = os.path.join(game_dir, game_files[game_idx])

            try:
                # Load the game
                game_history = torch.load(game_path)
                print(f"Loaded game with {len(game_history)} moves")

                # Simple analysis
                print(f"Game length: {len(game_history) - 1} moves")
                final_state = game_history[-1]

                winner = 1 if final_state.get_reward(1) > 0 else 2 if final_state.get_reward(1) < 0 else 0
                winner_str = "Player 1" if winner == 1 else "Player 2" if winner == 2 else "Draw"
                print(f"Game result: {winner_str}")

                # Option to visualize
                visualize = input("Visualize the game? (y/n): ").lower() == 'y'
                if visualize:
                    for i, state in enumerate(game_history):
                        display_board(state)
                        print(f"Move {i}/{len(game_history) - 1}")
                        input("Press Enter for next move...")

            except Exception as e:
                print(f"Error analyzing game: {e}")

        elif advanced_choice == "4":
            # Back to main menu
            main()
            return
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    except Exception as e:
        print(f"An error occurred: {e}")