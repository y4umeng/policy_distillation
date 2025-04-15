import argparse
import logging
import sys
from pathlib import Path

# Add the repository root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import configuration and training function
import wandb
from src.engine.cfg import CFG, show_cfg
from src.tools.train import main

def generate_data_sizes(subset="medium"):
    """Generate data sizes based on the specified subset size"""
    if subset == "small":
        return [100, 10000]
    elif subset == "medium":
        return [1, 10, 100, 1000, 10000]
    elif subset == "large":
        return [1, 10, 100, 1000, 5000, 10000, 50000, 100000]
    elif subset == "extras":
        return [500, 5000, 25000, 50000]
    else:  # all
        sizes = []
        # Powers of 10
        for i in range(0, 6):  # 1, 10, 100, 1000, 10000, 100000
            sizes.append(10**i)
        # Halfway points
        for i in range(0, 5):  # 5, 50, 500, 5000, 50000
            sizes.append(5 * 10**i)
        return sorted(sizes)

def generate_learning_rates(subset="medium"):
    """Generate learning rates based on the specified subset size"""
    if subset == "small":
        return [0, 0.1, 0.001, 0.00001]
    elif subset == "medium" or subset == "extras":
        return [0, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    elif subset == "large":
        return [0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 
                0.00005, 0.00001, 0.000005, 0.000001]
    else:  # all
        rates = [0]
        for power in range(0, -13, -1):  # 10^0 down to 10^-12
            rates.append(5 * 10**power)  # 0.5, 0.05, 0.005, ...
            rates.append(1 * 10**power)  # 0.1, 0.01, 0.001, ...
        return sorted(rates, reverse=True)

def train_model():
    """Training function that wandb will call for each hyperparameter combination"""
    # Initialize wandb run and get config
    with wandb.init() as run:
        # Get configuration from wandb
        config = wandb.config
        
        # Update CFG with the parameters from wandb
        CFG.DATA.MAX_CAPACITY = config.data_size
        CFG.DATA.INCREMENT = False
        CFG.DA.LR = config.learning_rate
        CFG.DA.PROB = config.prob
        CFG.DISTILLER.ENV = config.env
        CFG.DISTILLER.STUDENT = config.student
        CFG.EXPERIMENT.NAME = f"{config.student}_da_{config.env.replace('NoFrameskip-v4', '')}_{config.data_size}_{config.learning_rate}"
        CFG.LOG.FINAL_EVAL_EPS = config.eval_episodes
        CFG.SOLVER.EPOCHS = config.epochs
        CFG.LOG.EVAL_EPISODES = 0
        
        # Log the configuration
        logger.info(f"Running with configuration: {dict(config)}")
        
        try:
            # Run training with these hyperparameters
            main(CFG, False, [])
            logger.info(f"Completed run with data_size={config.data_size}, lr={config.learning_rate}")
        except Exception as e:
            logger.error(f"Error in run: {str(e)}")
            # Even if there's an error, wandb will still track it

def run_wandb_sweep():
    """Set up and run a wandb sweep"""
    parser = argparse.ArgumentParser("WandB Sweep for policy distillation")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--subset", type=str, choices=["small", "medium", "large", "all"], 
                        default="extras", help="Size of parameter sweep")
    parser.add_argument("--envs", type=str, nargs="+", default=["MsPacmanNoFrameskip-v4"],
                        help="Environments to test")
    parser.add_argument("--students", type=str, nargs="+", default=["dqn4"],
                        help="Student models to test")
    parser.add_argument("--prob", type=float, default=1.0, 
                        help="Probability value for DA.PROB")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--eval-episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")
    parser.add_argument("--project", type=str, default="rl_grid_sweeps",
                        help="rl_grid_sweeps")
    parser.add_argument("--entity", type=str, default=None,
                        help="WandB entity (username or team name)")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of runs to execute (None = unlimited)")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    global logger
    logger = logging.getLogger("wandb_sweep")
    
    # Load base configuration
    CFG.merge_from_file(args.config)
    
    # Get parameter sets
    data_sizes = generate_data_sizes(args.subset)
    learning_rates = generate_learning_rates(args.subset)
    
    logger.info(f"Setting up sweep with the following parameters:")
    logger.info(f"Data sizes: {data_sizes}")
    logger.info(f"Learning rates: {learning_rates}")
    logger.info(f"Environments: {args.envs}")
    logger.info(f"Student models: {args.students}")
    
    # Define wandb sweep configuration
    sweep_config = {
        "method": "grid",  # Using grid to ensure we try all combinations
        "metric": {
            "name": "best_score",
            "goal": "maximize"
        },
        "parameters": {
            "data_size": {
                "values": data_sizes
            },
            "learning_rate": {
                "values": learning_rates
            },
            "env": {
                "values": args.envs
            },
            "student": {
                "values": args.students
            },
            "prob": {
                "value": args.prob
            },
            "epochs": {
                "value": args.epochs
            },
            "eval_episodes": {
                "value": args.eval_episodes
            }
        }
    }
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=args.project,
        entity=args.entity
    )
    
    logger.info(f"Created sweep with ID: {sweep_id}")
    logger.info(f"Sweep URL: https://wandb.ai/{args.entity or 'your-username'}/{args.project}/sweeps/{sweep_id}")
    
    # Start the sweep agent
    wandb.agent(sweep_id, function=train_model, count=args.count)

if __name__ == "__main__":
    run_wandb_sweep()