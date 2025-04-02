#!/usr/bin/env python3
import os
import sys
import numpy as np

# Add necessary paths
sys.path.append('.')
sys.path.append('..')

# Import the study class and necessary modules
from transformer_study.transformer_study import SkudPaiShoTransformerStudy
# We need to import the state class for our patched method
from game.state import SkudPaiShoState


# Create a patched version of the class with a fixed evaluate_model method
class PatchedStudy(SkudPaiShoTransformerStudy):
    def evaluate_model(self, num_games=100, opponent="random"):
        """
        Patched version that handles None return from get_ai_move
        """
        print(f"Using patched evaluation method")

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
                    # Get model move - PATCHED to handle None return
                    try:
                        from ai_deprecated.utils import get_ai_move
                        result = get_ai_move(self.model, state)

                        if result is None:
                            # Fall back to random move if AI returns None
                            valid_moves = state.get_valid_moves()
                            if not valid_moves:
                                break
                            action = valid_moves[np.random.randint(len(valid_moves))]
                        else:
                            action, _ = result
                    except Exception as e:
                        print(f"Error in get_ai_move: {e}")
                        # Fall back to random move
                        valid_moves = state.get_valid_moves()
                        if not valid_moves:
                            break
                        action = valid_moves[np.random.randint(len(valid_moves))]
                else:  # Opponent's turn
                    if opponent == "random":
                        # Random opponent: choose a random valid move
                        valid_moves = state.get_valid_moves()
                        if not valid_moves:
                            break
                        action = valid_moves[np.random.randint(len(valid_moves))]

                    elif opponent == "self":
                        # Self-play: get another move from the model
                        try:
                            from ai_deprecated.utils import get_ai_move
                            result = get_ai_move(self.model, state)

                            if result is None:
                                valid_moves = state.get_valid_moves()
                                if not valid_moves:
                                    break
                                action = valid_moves[np.random.randint(len(valid_moves))]
                            else:
                                action, _ = result
                        except Exception as e:
                            print(f"Error in get_ai_move for self opponent: {e}")
                            valid_moves = state.get_valid_moves()
                            if not valid_moves:
                                break
                            action = valid_moves[np.random.randint(len(valid_moves))]

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


# Main execution
if __name__ == "__main__":
    # Initialize patched study
    study_name = 'continued_study_' + os.path.basename('transformer_study_20250304_124447')
    study = PatchedStudy(study_name=study_name)

    # Load your model
    print("Loading model...")
    model_path = 'C:/Users/kia/PycharmProjects/skud-pai-sho-ai/study_results/transformer_study_20250304_124447/transformer_model_iter_50.pth'
    study.load_model(model_path)

    try:
        # 2. Evaluating model against different opponents
        print('\n2. Evaluating model against different opponents...')
        for opponent in ['random', 'greedy', 'self']:
            try:
                print(f"Evaluating against {opponent}...")
                study.evaluate_model(num_games=50, opponent=opponent)
            except Exception as e:
                print(f"Error evaluating against {opponent}: {str(e)}")

        # 3. Performing policy analysis
        print('\n3. Performing policy analysis...')
        try:
            study.policy_analysis(num_games=20)
        except Exception as e:
            print(f"Error in policy analysis: {str(e)}")

        # 4. Performing attention analysis
        print('\n4. Performing attention analysis...')
        try:
            study.attention_analysis(num_samples=8)
        except Exception as e:
            print(f"Error in attention analysis: {str(e)}")

        # Save what we have so far
        study.save_study_results()

        # Continue with architecture study if previous steps succeeded
        print('\n5. Performing architecture study...')
        try:
            architectures = [
                {'d_model': 128, 'nhead': 4, 'num_layers': 3, 'dropout': 0.1},
                {'d_model': 256, 'nhead': 8, 'num_layers': 6, 'dropout': 0.1},
                {'d_model': 512, 'nhead': 8, 'num_layers': 6, 'dropout': 0.1},
                {'d_model': 256, 'nhead': 4, 'num_layers': 9, 'dropout': 0.1},
                {'d_model': 256, 'nhead': 8, 'num_layers': 3, 'dropout': 0.2}
            ]
            study.architecture_study(architectures, iterations=10, games_per_iteration=10)
        except Exception as e:
            print(f"Error in architecture study: {str(e)}")

        # Training efficiency study
        print('\n6. Performing training efficiency study...')
        try:
            training_configs = [
                {'iterations': 20, 'games_per_iteration': 10, 'epochs_per_iteration': 3},
                {'iterations': 10, 'games_per_iteration': 20, 'epochs_per_iteration': 3},
                {'iterations': 5, 'games_per_iteration': 40, 'epochs_per_iteration': 3},
                {'iterations': 20, 'games_per_iteration': 10, 'epochs_per_iteration': 6},
                {'iterations': 40, 'games_per_iteration': 5, 'epochs_per_iteration': 3}
            ]
            study.training_efficiency_study(training_configs)
        except Exception as e:
            print(f"Error in training efficiency study: {str(e)}")

        # Q-learning comparison
        try:
            q_agent_path = input('Enter path to Q-learning agent (or press Enter to skip): ')
            if q_agent_path:
                print('\n7. Comparing with Q-learning agent...')
                study.compare_with_q_learning(q_agent_path)
        except Exception as e:
            print(f"Error in Q-learning comparison: {str(e)}")

    except Exception as e:
        print(f"Critical error: {str(e)}")
    finally:
        # Always save results at the end
        print('\nSaving study results...')
        study.save_study_results()
        print("Study completed or terminated.")