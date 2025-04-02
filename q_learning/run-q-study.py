#!/usr/bin/env python3
# run_study.py - Script to run a comprehensive study of Q-learning for Skud Pai Sho

import os
import sys
import argparse
from datetime import datetime
from q_study_framework import SkudPaiShoStudy


def main():
    parser = argparse.ArgumentParser(description='Run a Q-learning study for Skud Pai Sho')
    parser.add_argument('--study-name', default=f'skud_study_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        help='Name of the study (default: auto-generated timestamp)')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of episodes for main training (default: 10000)')
    parser.add_argument('--parameter-study', action='store_true',
                        help='Run parameter study with different combinations')
    parser.add_argument('--q-only', action='store_true',
                        help='Study Q-learning only (skip transformer comparison)')
    args = parser.parse_args()

    print(f"Starting Skud Pai Sho Q-learning study: {args.study_name}")
    print(f"Training for {args.episodes} episodes")

    # Create study
    study = SkudPaiShoStudy(study_name=args.study_name)

    # Define parameter combinations if requested
    parameter_combinations = None
    if args.parameter_study:
        parameter_combinations = [
            # Different learning rates
            {"learning_rate": 0.01, "discount_factor": 0.9, "exploration_rate": 1.0,
             "exploration_decay": 0.995, "min_exploration_rate": 0.01},
            {"learning_rate": 0.1, "discount_factor": 0.9, "exploration_rate": 1.0,
             "exploration_decay": 0.995, "min_exploration_rate": 0.01},
            {"learning_rate": 0.2, "discount_factor": 0.9, "exploration_rate": 1.0,
             "exploration_decay": 0.995, "min_exploration_rate": 0.01},

            # Different discount factors
            {"learning_rate": 0.1, "discount_factor": 0.8, "exploration_rate": 1.0,
             "exploration_decay": 0.995, "min_exploration_rate": 0.01},
            {"learning_rate": 0.1, "discount_factor": 0.95, "exploration_rate": 1.0,
             "exploration_decay": 0.995, "min_exploration_rate": 0.01},
            {"learning_rate": 0.1, "discount_factor": 0.99, "exploration_rate": 1.0,
             "exploration_decay": 0.995, "min_exploration_rate": 0.01},

            # Different exploration decay rates
            {"learning_rate": 0.1, "discount_factor": 0.9, "exploration_rate": 1.0,
             "exploration_decay": 0.99, "min_exploration_rate": 0.01},
            {"learning_rate": 0.1, "discount_factor": 0.9, "exploration_rate": 1.0,
             "exploration_decay": 0.98, "min_exploration_rate": 0.01},
            {"learning_rate": 0.1, "discount_factor": 0.9, "exploration_rate": 1.0,
             "exploration_decay": 0.999, "min_exploration_rate": 0.01},
        ]

        print(f"Running parameter study with {len(parameter_combinations)} combinations")

    # Run modified full study
    if args.q_only:
        # Initialize Q-learning agent
        print("\n1. Training basic Q-learning agent...")
        study.initialize_q_agent()
        study.train_q_agent(num_episodes=args.episodes, print_interval=args.episodes // 10)

        # Perform state space analysis
        print("\n2. Performing state space analysis...")
        study.state_space_analysis(num_games=100)

        # Perform action analysis
        print("\n3. Performing action analysis...")
        study.action_analysis(num_games=50)

        # Parameter study (if parameter combinations provided)
        if parameter_combinations:
            print("\n4. Performing parameter study...")
            study.parameter_study(parameter_combinations, episodes_per_combo=args.episodes // 10)

        # Save all results
        print("\n5. Saving study results...")
        study.save_study_results()
    else:
        # Run full study including transformer comparison
        study.run_full_study(num_episodes=args.episodes, parameter_combinations=parameter_combinations)

    print(f"\nStudy completed: {args.study_name}")
    print(f"Results saved in: study_results/{args.study_name}/")


if __name__ == "__main__":
    main()
