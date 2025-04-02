#!/usr/bin/env python3
# run_transformer_study.py - Script to run a comprehensive study of the transformer model for Skud Pai Sho

import os
import sys
import argparse
from datetime import datetime
from transformer_study import SkudPaiShoTransformerStudy


def main():
    parser = argparse.ArgumentParser(description='Run a transformer model study for Skud Pai Sho')
    parser.add_argument('--study-name', default=f'transformer_study_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        help='Name of the study (default: auto-generated timestamp)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of training iterations (default: 50)')
    parser.add_argument('--games', type=int, default=20,
                        help='Number of games per iteration (default: 20)')
    parser.add_argument('--quick', action='store_true',
                        help='Run a quick study with fewer iterations and games')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to a pre-trained model to start with')
    parser.add_argument('--skip-arch', action='store_true',
                        help='Skip architecture study')
    parser.add_argument('--skip-efficiency', action='store_true',
                        help='Skip training efficiency study')
    parser.add_argument('--skip-q-comparison', action='store_true',
                        help='Skip comparison with Q-learning')
    args = parser.parse_args()

    print(f"Starting Skud Pai Sho transformer study: {args.study_name}")

    # Adjust parameters for quick study
    if args.quick:
        args.iterations = 10
        args.games = 10
        print("Running quick study with reduced iterations and games")

    print(f"Training for {args.iterations} iterations with {args.games} games per iteration")

    # Create study
    study = SkudPaiShoTransformerStudy(study_name=args.study_name)

    # Load pre-trained model if specified
    if args.load_model:
        print(f"Loading pre-trained model from {args.load_model}")
        study.load_model(args.load_model)

    # Run full study with specified parameters
    study.run_full_study(
        iterations=args.iterations,
        games_per_iteration=args.games,
        arch_study=not args.skip_arch,
        efficiency_study=not args.skip_efficiency,
        compare_q=not args.skip_q_comparison
    )

    print(f"\nStudy completed: {args.study_name}")
    print(f"Results saved in: study_results/{args.study_name}/")


if __name__ == "__main__":
    main()