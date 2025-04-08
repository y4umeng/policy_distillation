import torch.backends.cudnn as cudnn
import wandb
import argparse
import sys
from pathlib import Path

# Set CUDA benchmark for performance
cudnn.benchmark = True

# Add the repository root to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

# Import configuration and training function
from src.engine.cfg import CFG as cfg
from src.tools.train import main

def run_sweep():
    # Initialize wandb for this run
    wandb.init()
    
    # Update configuration with wandb sweep parameters
    cfg.DA.LR = wandb.config.lr
    cfg.DA.PROB = wandb.config.prob
    cfg.LOG.FINAL_EVAL_EPS = 0
    
    # Train with the updated configuration
    main(cfg, False, [])

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser("Hyperparameter sweep for policy distillation")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--count", type=int, default=100, help="Number of sweep runs to perform")
    parser.add_argument("--project", type=str, default="policy_distillation_sweeps", help="Wandb project name")
    args = parser.parse_args()
    
    # Load the specified configuration file
    cfg.merge_from_file(args.config)
    
    # Define the sweep configuration
    sweep_configuration = {
        "method": "bayes",
        "metric": {"goal": "maximize", "name": "best_score"},
        "parameters": {
            "lr": {"max": 0.01, "min": 0.00001, "distribution": "log_uniform_values"},
            "prob": {"max": 0.9, "min": 0.1}
        },
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 10,
            "eta": 2
        }
    }
    
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)
    
    # Start the sweep agent
    wandb.agent(sweep_id, function=run_sweep, count=args.count)