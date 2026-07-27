# Skud Pai Sho AI System

This system implements an AI for playing the game of Skud Pai Sho using modern deep learning techniques, specifically a transformer-based model combined with Monte Carlo Tree Search (MCTS). The system includes tools for training, playing against the AI, and analyzing model behavior.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
   - [Training a New Model](#training-a-new-model)
   - [Playing Against an AI](#playing-against-an-ai)
   - [Evaluating Model Performance](#evaluating-model-performance)
5. [Advanced Features](#advanced-features)
   - [Opening Book](#opening-book)
   - [Visualizing Attention](#visualizing-attention)
   - [Comparing Models](#comparing-models)
   - [Q-Learning Alternative](#q-learning-alternative)
6. [Command Line Arguments](#command-line-arguments)
7. [Game Rules](#game-rules)
8. [Troubleshooting](#troubleshooting)

## System Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Matplotlib
- tqdm (for progress bars)

## Project Structure

The project is organized into several main components:

- **game/**: Implements the rules and state representation of Skud Pai Sho
  - `state.py`: Core game logic and state representation
  - `display.py`: Functions for displaying the board in ASCII format

- **ai/**: AI implementation and training logic
  - `model.py`: Implementation of the transformer-based neural network
  - `mcts.py`: Monte Carlo Tree Search implementation
  - `integrated_training.py`: Main training loop with MCTS
  - `opening_book.py`: Opening book implementation for storing common openings
  - `training.py`: Utility functions for training
  - `utils.py`: Helper functions for AI evaluation and gameplay

- **scripts**: Various utility scripts
  - `train_improved_mcts.py`: Script for training a model with MCTS
  - `evaluate.py`: Script for evaluating model performance
  - `analyze.py`: Script for analyzing specific positions
  - `visualize_attention.py`: Script for visualizing model attention
  - `compare_models.py`: Script for comparing different models
  - `main.py`: Main entry point for playing against the AI

- **q_learning_ai.py**: Alternative implementation using Q-learning

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/skud-pai-sho-ai.git
   cd skud-pai-sho-ai
   ```

2. Install requirements:
   ```
   pip install torch numpy matplotlib tqdm
   ```

## Basic Usage

### Training a New Model

The easiest way to train a new model is using the `train_improved_mcts.py` script:

```
python train_improved_mcts.py --iterations 50 --games 20 --mcts_sims 800 --epochs 5
```

This will train a model for 50 iterations, with 20 self-play games per iteration, using 800 MCTS simulations per move, and 5 training epochs per iteration. Models will be saved in the `models/` directory.

Alternatively, you can use the interactive interface:

```
python main.py
```

Then select option 1 to train a new model.

### Playing Against an AI

To play against a trained model:

```
python main.py
```

Select option 2 to play against a pre-trained model, or option 3 to play against a randomly initialized model.

### Evaluating Model Performance

To evaluate a model's performance:

```
python evaluate.py --model_path models/your_model.pth --num_games 50 --mcts_sims 100
```

This will evaluate the model over 50 games, using 100 MCTS simulations per move, and report win rates and other statistics.

## Advanced Features

### Opening Book

The system supports maintaining an opening book to improve play in the early stages of the game:

```
python train_improved_mcts.py --use_opening_book --opening_book_file data/opening_book.json
```

To initialize a sample opening book:

```
python main.py
```

Then select option 4 (Advanced options) and then option 1 (Create initial opening book).

### Visualizing Attention

To visualize how the model's attention mechanism focuses on different parts of the board:

```
python visualize_attention.py --model_path models/your_model.pth --mode game --num_moves 10
```

This will play a game and generate visualizations of the attention patterns for each move.

### Comparing Models

To compare the performance of different models:

```
python compare_models.py --transformer_model models/model1.pth --q_model models/q_agent.pkl --num_games 50
```

This will pit the transformer model against a Q-learning model and report comparative statistics.

### Q-Learning Alternative

The system also includes an alternative implementation using Q-learning:

```
python run-q-study.py --study-name my_study --episodes 10000
```

This will train a Q-learning agent and conduct a study of its performance.

## Command Line Arguments

### Training (train_improved_mcts.py)

- `--iterations`: Number of training iterations
- `--games`: Number of self-play games per iteration
- `--mcts_sims`: Number of MCTS simulations per move
- `--epochs`: Training epochs per iteration
- `--model_dir`: Directory to save models
- `--log_dir`: Directory to save logs
- `--load_model`: Path to load initial model
- `--architecture`: Model architecture (A1-A5)
- `--use_opening_book`: Use opening book
- `--opening_book_file`: Opening book file
- `--debug`: Enable debug output and reduced settings

### Evaluation (evaluate.py)

- `--model_path`: Path to model file
- `--opponent_model`: Path to opponent model file
- `--num_games`: Number of games to play
- `--mcts_sims`: Number of MCTS simulations per move
- `--opponent_type`: Type of opponent (random, mcts, model)
- `--use_opening_book`: Use opening book
- `--opening_book_file`: Opening book file
- `--output_dir`: Directory to save evaluation results
- `--verbose`: Print detailed evaluation progress
- `--save_games`: Save game histories

### Attention Visualization (visualize_attention.py)

- `--model_path`: Path to model file
- `--mode`: Visualization mode (game or position)
- `--num_moves`: Number of moves to play and visualize
- `--mcts_sims`: Number of MCTS simulations per move
- `--player_side`: Which player you want to play as (1 or 2)
- `--moves`: Comma-separated list of moves to reach position
- `--game_file`: Game history file to load
- `--move_number`: Move number to analyze from game file
- `--output_dir`: Directory to save visualizations
- `--no_show`: Do not display visualizations interactively

## Game Rules

Skud Pai Sho is a strategic board game played on a circular board with various tile types. The goal is to create harmonies between your tiles.

### Basic Rules:
- Players take turns planting and moving tiles
- The first player to form four harmonies wins
- A harmony is formed when two flower tiles of the same player are in line with each other
- Different tile types have different movement capabilities

For a full ruleset, please refer to a comprehensive Skud Pai Sho guide.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce the number of MCTS simulations or use a smaller model architecture.

2. **Model not learning**: Try increasing the number of games per iteration or the number of epochs.

3. **Slow training**: Reduce MCTS simulations, use CPU instead of GPU, or try the Q-learning approach which is less computationally intensive.

4. **Opening book errors**: Make sure the opening book file exists and is valid JSON. You can create a new one using the provided utilities.

5. **Input/output channel mismatch**: Ensure your model architecture matches the state encoding dimensions. The system should automatically detect the correct dimensions.

If you encounter other issues, please check the model and training parameters, or report an issue to the project repository.
