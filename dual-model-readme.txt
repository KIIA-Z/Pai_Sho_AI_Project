# Enhanced Skud Pai Sho AI Evaluation Framework

This framework provides a comprehensive set of tools for evaluating and comparing both transformer-based neural network models and Q-learning agents for Skud Pai Sho. It generates meaningful metrics across multiple dimensions of gameplay and strategy, with special features for analyzing the differences between model types.

## Installation

1. Place the following files in your Skud Pai Sho project directory:
   - `skud_pai_sho_evaluation_enhanced.py` - The enhanced dual-model evaluation framework
   - `evaluate_skud_pai_sho_enhanced.py` - Command-line script for running evaluations

2. Make sure you have the required dependencies:
   ```
   pip install numpy pandas matplotlib seaborn tqdm
   ```

## Basic Usage

To evaluate transformer models against Q-learning agents:

```bash
python evaluate_skud_pai_sho_enhanced.py \
  --transformer_models models/transformer_model.pth \
  --q_learning_models models/q_learning_model.pkl \
  --model_names "Transformer A" "Q-Learning A"
```

## Features

The enhanced framework provides the following capabilities:

### Cross-Model Evaluation
- Direct comparison between transformer and Q-learning models
- Fair evaluation adapting to each model's action selection method
- Distinct analysis of each model type's strengths and weaknesses

### Win Rate Analysis
- Comprehensive breakdown of win/loss/draw rates for each model
- Head-to-head matchup matrix
- Performance metrics by model type

### Computational Efficiency Analysis
- Comparison of thinking time across model types
- Statistical breakdown of computational performance

### Harmony Analysis
- Tracking of harmony progression throughout games
- Analysis of harmony generation by model type
- Correlations between harmony strategies and outcomes

### Game Length Analysis
- Distribution of game lengths by model type
- Analysis of game length patterns in different matchups (transformer vs Q-learning)

### Move Sequence Analysis
- Separate analysis of winning sequences for each model type
- Comparison of distinctive strategies between approaches

## Command-Line Options

```
--transformer_models PATH   Paths to transformer model files
--q_learning_models PATH    Paths to Q-learning model files
--model_names NAME          Names for the models (optional)
--num_games N               Number of games to generate for each matchup (default: 20)
--mcts_sims N               Number of MCTS simulations per move (default: 100)
--use_games_from PATH       Load previously played games instead of generating new ones
--use_opening_book          Use opening book for move selection
--opening_book_file PATH    Path to opening book file (default: data/opening_book.json)
--output_dir DIR            Directory to save results (default: evaluation_results)
--verbose                   Print detailed progress information
```

## Output

The framework generates several outputs:

1. `game_data.json` - Raw game data for further analysis
2. `evaluation_summary.json` - Summary of all evaluation results
3. `evaluation_report.html` - Interactive HTML report with visualizations
4. Various analysis-specific JSON files and visualizations

## Example

```bash
# Compare transformer and Q-learning models with custom names
python evaluate_skud_pai_sho_enhanced.py \
  --transformer_models models/transformer_v1.pth models/transformer_v2.pth \
  --q_learning_models models/q_learning_alpha.pkl models/q_learning_beta.pkl \
  --model_names "T-Model 1" "T-Model 2" "Q-Agent Alpha" "Q-Agent Beta" \
  --num_games 30 \
  --mcts_sims 200 \
  --output_dir results/transformer_vs_qlearning \
  --verbose
```

## Implementation Notes

The framework makes special accommodations for each model type:

### Transformer Models
- Uses MCTS for move selection
- Properly handles policy and value outputs
- Works with the existing neural network architecture

### Q-Learning Models
- Uses the Q-learning agent's native action selection
- Temporarily disables exploration for deterministic evaluation
- Properly loads models saved with pickle

### HTML Report

The HTML report includes:
- Color-coded identification of model types
- Separate visualizations for each model type
- Direct comparisons between approaches

## Requirements

- Python 3.6+
- PyTorch
- NumPy, Pandas
- Matplotlib, Seaborn
- Skud Pai Sho game implementation
- Both transformer model and Q-learning agent implementations