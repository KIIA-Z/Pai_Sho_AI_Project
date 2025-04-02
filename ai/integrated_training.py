# ai/integrated_training.py
import torch
import numpy as np
import os
import time
from datetime import datetime
import json
import random

# Import your existing modules
from game.state import SkudPaiShoState, TileType, BOARD_SIZE
from ai.model import SkudPaiShoTransformer
from ai.training import action_to_index, index_to_action

# Import new modules
from ai.mcts import mcts_search, self_play_with_mcts, set_state_class
from ai.opening_book import OpeningBook, get_move_with_opening_book_and_mcts


def train_skud_pai_sho_ai_with_mcts(
        model,
        iterations=50,
        games_per_iteration=20,
        mcts_simulations=800,
        epochs_per_iteration=5,
        start_lr=0.001,
        min_lr=0.00001,
        weight_decay=1e-4,
        patience=3,
        eval_every=5,
        temperature_schedule=None,
        use_opening_book=True,
        opening_book_file="data/opening_book.json",
        model_dir="models",
        log_dir="logs"
):
    """
    Main training loop with MCTS and opening book integration.

    Args:
        model: The neural network model
        iterations: Number of iterations to train
        games_per_iteration: Number of self-play games per iteration
        mcts_simulations: Number of MCTS simulations per move
        epochs_per_iteration: Training epochs per iteration
        start_lr: Initial learning rate
        min_lr: Minimum learning rate
        weight_decay: L2 regularization parameter
        patience: Number of iterations to wait before reducing learning rate
        eval_every: How often to evaluate and save best model
        temperature_schedule: Dict mapping move number to temperature
        use_opening_book: Whether to use an opening book
        opening_book_file: Path to opening book file
        model_dir: Directory to save models
        log_dir: Directory to save logs

    Returns:
        Trained model and training statistics
    """
    import torch.optim as optim
    import torch.optim.lr_scheduler as lr_scheduler
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F

    # Set the state class for MCTS
    set_state_class(SkudPaiShoState)

    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize opening book if enabled
    opening_book = None
    if use_opening_book:
        opening_book = OpeningBook(book_file=opening_book_file)

    # Default temperature schedule if none provided
    if temperature_schedule is None:
        temperature_schedule = {0: 1.0, 20: 0.5, 40: 0.25}

    # For tracking the best model
    best_win_rate = 0.0
    best_model_state = None
    no_improvement_count = 0

    # Dynamic learning rate based on performance
    current_lr = start_lr

    # Keep track of metrics
    policy_losses = []
    value_losses = []
    win_rates = []
    game_lengths = []
    harmony_counts = []

    # Training log
    training_log = {
        "iterations": [],
        "policy_losses": [],
        "value_losses": [],
        "win_rates": [],
        "game_lengths": [],
        "harmony_counts": [],
        "learning_rates": []
    }

    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/{iterations}")
        print(f"Current learning rate: {current_lr:.6f}")

        # Generate self-play games using MCTS
        print(f"Generating {games_per_iteration} self-play games with MCTS...")
        start_time = time.time()

        # UPDATED GAME GENERATION LOOP STARTS HERE
        game_records = []
        for game_num in range(games_per_iteration):
            state = SkudPaiShoState()
            game_history = []
            move_num = 0

            print(f"Game {game_num + 1}/{games_per_iteration}", end="\r")

            # Add time tracking for each game
            start_time_game = time.time()
            max_game_time = 300  # 5 minutes per game
            seen_states = set()  # For detecting repeating states

            while not state.is_game_over() and move_num < 200:  # Max 200 moves
                # Print detailed progress every 10 moves
                if move_num % 10 == 0:
                    print(f"Game {game_num + 1}/{games_per_iteration}, Move {move_num}, Player {state.current_player}")

                # Check if game has been running too long
                if time.time() - start_time_game > max_game_time:
                    print(f"Game {game_num + 1} timed out after {time.time() - start_time_game:.1f} seconds")
                    break

                # Detect repeating states to avoid cycles
                state_hash = hash(str(state.board.tobytes()))
                if state_hash in seen_states:
                    print(f"Game {game_num + 1} detected repeating state at move {move_num}, ending game")
                    break
                seen_states.add(state_hash)

                # Determine temperature based on move number
                temperature = 1.0
                for threshold, temp in sorted(temperature_schedule.items()):
                    if move_num >= threshold:
                        temperature = temp

                # Use opening book + MCTS to select move and get improved policy
                if use_opening_book and opening_book is not None:
                    move, policy = get_move_with_opening_book_and_mcts(
                        model,
                        state,
                        opening_book,
                        mcts_simulations=mcts_simulations,
                        temperature=temperature
                    )
                else:
                    # Use pure MCTS
                    move, policy = mcts_search(
                        model,
                        state,
                        num_simulations=mcts_simulations,
                        temperature=temperature,
                        dirichlet_noise=True
                    )

                if move is None:
                    print(f"Game {game_num + 1}: No valid moves at move {move_num}, ending game")
                    break  # No valid moves

                # If we got a move from the opening book without a policy,
                # we need to generate a policy using MCTS
                if policy is None:
                    _, policy = mcts_search(
                        model,
                        state,
                        num_simulations=mcts_simulations // 2,  # Half the usual sims for efficiency
                        temperature=temperature
                    )

                # Store state and improved policy
                game_history.append((state.copy(), policy))

                # Make move
                state.make_move(move)
                move_num += 1

            # Game over, determine outcome
            game_end_reason = "completed normally"
            if move_num >= 200:
                game_end_reason = "reached move limit"
            elif time.time() - start_time_game > max_game_time:
                game_end_reason = "timed out"
            elif state_hash in seen_states:
                game_end_reason = "detected cycle"
            elif move is None:
                game_end_reason = "no valid moves"

            print(
                f"Game {game_num + 1} {game_end_reason} after {move_num} moves in {time.time() - start_time_game:.1f} seconds")

            # Variable to store player1_reward
            player1_reward = 0

            if state.is_game_over():
                player1_reward = state.get_reward(1)

                # Add outcome to all states in game
                for past_state, policy in game_history:
                    # Value target is from perspective of player who just moved
                    player_reward = player1_reward if past_state.current_player == 1 else -player1_reward
                    game_records.append((past_state, policy, player_reward))

            # Add game statistics if this is the end of the game
            if state.is_game_over() or move_num >= 200:
                game_lengths.append(move_num)
                harmony_counts.append(state.count_harmonies())

                # Add game outcome for win rate calculation
                if state.is_game_over():
                    if player1_reward > 0:
                        win_rates.append(1)
                    elif player1_reward < 0:
                        win_rates.append(0)
                    else:
                        win_rates.append(0.5)  # Draw
                else:
                    win_rates.append(0.5)  # Timeout = draw
        # UPDATED GAME GENERATION LOOP ENDS HERE

        generation_time = time.time() - start_time
        print(f"Generated {len(game_records)} training examples in {generation_time:.1f} seconds")

        # Learn from these games for the opening book
        if use_opening_book and opening_book is not None:
            opening_book.learn_from_games(game_records)
            opening_book.save()
            opening_stats = opening_book.get_stats()
            print(f"Opening book now has {opening_stats['total_positions']} positions")

        # Create a training dataset from game records
        states = []
        values = []
        policies = []

        for state, policy, value in game_records:
            states.append(state.encode_for_network())
            policies.append(policy)
            values.append(value)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        values = torch.tensor(np.array(values), dtype=torch.float32)
        policies = torch.tensor(np.array(policies), dtype=torch.float32)

        # Create data loader
        dataset = TensorDataset(states, policies, values)
        data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

        # Optimizer with weight decay (L2 regularization)
        optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=weight_decay)

        # Learning rate scheduler: Cosine annealing with warm restarts
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=epochs_per_iteration,
            T_mult=1,
            eta_min=current_lr / 10
        )

        # Train model on new data
        model.train()
        device = next(model.parameters()).device

        running_policy_loss = 0.0
        running_value_loss = 0.0

        for epoch in range(epochs_per_iteration):
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0

            for batch_idx, (state_batch, policy_batch, value_batch) in enumerate(data_loader):
                # Move to device
                state_batch = state_batch.to(device)
                policy_batch = policy_batch.to(device)
                value_batch = value_batch.to(device)

                # Forward pass
                policy_output, value_output = model(state_batch)

                # Calculate losses
                policy_loss = F.kl_div(
                    F.log_softmax(policy_output, dim=1),
                    policy_batch,
                    reduction='batchmean'
                )
                value_loss = F.mse_loss(value_output.squeeze(), value_batch)

                # Combined loss with emphasis on value
                loss = policy_loss + 1.5 * value_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Update running losses
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()

            # Step the scheduler at the end of each epoch
            scheduler.step()

            # Calculate average losses for this epoch
            avg_policy_loss = epoch_policy_loss / len(data_loader)
            avg_value_loss = epoch_value_loss / len(data_loader)

            print(f"Epoch {epoch + 1}/{epochs_per_iteration}, "
                  f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")

        # Update loss tracking
        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)

        # Save checkpoint
        checkpoint_path = os.path.join(model_dir, f"model_iter_{iteration}.pth")
        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
        }, checkpoint_path)

        # Calculate average game statistics
        if game_lengths:
            avg_game_length = sum(game_lengths) / len(game_lengths)
        else:
            avg_game_length = 0

        if harmony_counts:
            avg_harmony_count = sum(harmony_counts) / len(harmony_counts)
        else:
            avg_harmony_count = 0

        if win_rates:
            current_win_rate = sum(win_rates) / len(win_rates)
        else:
            current_win_rate = 0

        print(f"Avg Game Length: {avg_game_length:.1f}, "
              f"Avg Harmony Count: {avg_harmony_count:.2f}, "
              f"Win Rate: {current_win_rate:.3f}")

        # Update training log
        training_log["iterations"].append(iteration)
        training_log["policy_losses"].append(avg_policy_loss)
        training_log["value_losses"].append(avg_value_loss)
        training_log["win_rates"].append(current_win_rate)
        training_log["game_lengths"].append(avg_game_length)
        training_log["harmony_counts"].append(avg_harmony_count)
        training_log["learning_rates"].append(current_lr)

        # Save training log
        with open(os.path.join(log_dir, "training_log.json"), "w") as f:
            json.dump(training_log, f, indent=2)

        # Periodic evaluation against fixed opponent
        if iteration % eval_every == 0 or iteration == iterations - 1:
            print("Evaluating model...")

            # Create a deterministic evaluation copy of the model
            eval_model = type(model)(
                input_channels=model.conv1.in_channels,
                d_model=model.transformer_encoder.layers[0].self_attn.embed_dim,
                nhead=model.transformer_encoder.layers[0].self_attn.num_heads,
                num_layers=len(model.transformer_encoder.layers),
                dropout=0.0  # No dropout for evaluation
            )
            eval_model.load_state_dict(model.state_dict())
            eval_model.eval()
            eval_model = eval_model.to(device)

            # Play against previous best model or random if no best model yet
            opponent_model = None
            if best_model_state is not None:
                opponent_model = type(model)(
                    input_channels=model.conv1.in_channels,
                    d_model=model.transformer_encoder.layers[0].self_attn.embed_dim,
                    nhead=model.transformer_encoder.layers[0].self_attn.num_heads,
                    num_layers=len(model.transformer_encoder.layers),
                    dropout=0.0
                )
                opponent_model.load_state_dict(best_model_state)
                opponent_model.eval()
                opponent_model = opponent_model.to(device)

            # Evaluate against opponent
            eval_wins = 0
            eval_losses = 0
            eval_draws = 0
            num_eval_games = 10

            for game_num in range(num_eval_games):
                state = SkudPaiShoState()

                while not state.is_game_over() and state.turn_number < 200:
                    if state.current_player == 1:  # Our model's turn
                        if use_opening_book and opening_book is not None:
                            move, _ = get_move_with_opening_book_and_mcts(
                                eval_model,
                                state,
                                opening_book,
                                mcts_simulations=mcts_simulations // 2,
                                temperature=0.0  # Deterministic for evaluation
                            )
                        else:
                            move, _ = mcts_search(
                                eval_model,
                                state,
                                num_simulations=mcts_simulations // 2,
                                temperature=0.0,
                                dirichlet_noise=False
                            )
                    else:  # Opponent's turn
                        if opponent_model:
                            move, _ = mcts_search(
                                opponent_model,
                                state,
                                num_simulations=mcts_simulations // 2,
                                temperature=0.0,
                                dirichlet_noise=False
                            )
                        else:
                            # Random opponent
                            valid_moves = state.get_valid_moves()
                            if not valid_moves:
                                break
                            move = random.choice(valid_moves)

                    if move is None:
                        break

                    state.make_move(move)

                # Game over, determine outcome
                if state.is_game_over():
                    result = state.get_reward(1)  # From player 1's perspective
                    if result > 0:
                        eval_wins += 1
                    elif result < 0:
                        eval_losses += 1
                    else:
                        eval_draws += 1
                else:
                    # Draw by move limit
                    eval_draws += 1

            eval_win_rate = eval_wins / num_eval_games
            print(f"Evaluation results: Wins: {eval_wins}, Losses: {eval_losses}, "
                  f"Draws: {eval_draws}, Win Rate: {eval_win_rate:.3f}")

            # Save the best model
            if eval_win_rate > best_win_rate:
                best_win_rate = eval_win_rate
                best_model_state = model.state_dict().copy()
                no_improvement_count = 0

                # Save best model
                best_model_path = os.path.join(model_dir, "best_model.pth")
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': best_model_state,
                    'win_rate': best_win_rate,
                }, best_model_path)

                print(f"New best model with win rate: {best_win_rate:.3f}")
            else:
                no_improvement_count += 1
                print(f"No improvement for {no_improvement_count} evaluations")

            # Learning rate adjustment based on performance
            if no_improvement_count >= patience:
                # Reduce learning rate
                current_lr = max(current_lr * 0.5, min_lr)
                print(f"Reducing learning rate to {current_lr:.6f}")
                no_improvement_count = 0

                # If we're using the best model, load it back
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print("Reverting to best model and continuing training")

    # At the end, use the best model found
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Training complete. Using best model with win rate: {best_win_rate:.3f}")

    # Final save
    final_model_path = os.path.join(model_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'win_rate': best_win_rate,
        'training_log': training_log
    }, final_model_path)

    # Return training statistics as well as the model
    return model, training_log