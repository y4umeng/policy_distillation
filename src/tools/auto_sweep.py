import argparse
import itertools
import math
import logging
from pathlib import Path
import sys

# Add the repository root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import configuration and training function
from src.engine.cfg import CFG, show_cfg
from src.tools.train import main

def generate_data_sizes(subset="large"):
    """Generate data sizes based on the specified subset size"""
    if subset == "small":
        return [100, 10000]
    elif subset == "medium":
        return [1, 10, 100, 1000, 10000, 100000]
    elif subset == "large":
        return [1, 10, 100, 1000, 5000, 10000, 50000, 100000]
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
        return [0.1, 0.001, 0.00001]
    elif subset == "medium":
        return [0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    elif subset == "large":
        return [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 
                0.00005, 0.00001, 0.000005, 0.000001]
    else:  # all
        rates = []
        for power in range(0, -13, -1):  # 10^0 down to 10^-12
            rates.append(5 * 10**power)  # 0.5, 0.05, 0.005, ...
            rates.append(1 * 10**power)  # 0.1, 0.01, 0.001, ...
        return sorted(rates, reverse=True)

def run_sweep():
    """Run a parameter sweep over data sizes and learning rates"""
    parser = argparse.ArgumentParser("Parameter sweep for policy distillation")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--subset", type=str, choices=["small", "medium", "large", "all"], 
                        default="medium", help="Size of parameter sweep")
    parser.add_argument("--envs", type=str, nargs="+", default=["MsPacmanNoFrameskip-v4"],
                        help="Environments to test")
    parser.add_argument("--students", type=str, nargs="+", default=["dqn4"],
                        help="Student models to test")
    parser.add_argument("--prob", type=float, default=1.0, 
                        help="Probability value for DA.PROB")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--eval-episodes", type=int, default=1000,
                        help="Number of evaluation episodes")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("sweep")
    
    # Load base configuration
    CFG.merge_from_file(args.config)
    
    # Generate parameter combinations
    data_sizes = generate_data_sizes(args.subset)
    learning_rates = generate_learning_rates(args.subset)
    
    # Get all combinations
    combinations = list(itertools.product(data_sizes, learning_rates, args.envs, args.students))
    total_runs = len(combinations)
    
    logger.info(f"Starting parameter sweep with {total_runs} combinations")
    logger.info(f"Data sizes: {data_sizes}")
    logger.info(f"Learning rates: {learning_rates}")
    logger.info(f"Environments: {args.envs}")
    logger.info(f"Student models: {args.students}")
    
    # Run each combination
    for run_index, (data_size, lr, env, student) in enumerate(combinations):
        # Format LR with appropriate precision
        lr_str = f"{lr:.12g}"
        
        # Create a unique experiment name
        env_short = env.replace("NoFrameskip-v4", "")
        run_name = f"{student}_da_{env_short}_{data_size}_{lr_str}".replace('-', 'neg').replace('.', 'p')
        
        logger.info(f"Run {run_index+1}/{total_runs}: data_size={data_size}, lr={lr_str}, env={env}, student={student}")
        
        # Update configuration
        CFG.DATA.MAX_CAPACITY = data_size
        CFG.DATA.INCREMENT = False
        CFG.DA.LR = lr
        CFG.DA.PROB = args.prob
        CFG.DISTILLER.ENV = env
        CFG.DISTILLER.STUDENT = student
        CFG.EXPERIMENT.NAME = run_name
        CFG.LOG.FINAL_EVAL_EPS = args.eval_episodes
        CFG.SOLVER.EPOCHS = args.epochs
        
        # Display the configuration
        logger.info(f"Configuration for run: {run_name}")
        show_cfg(CFG)
        
        try:
            # Run training with this configuration
            main(CFG, False, [])
            logger.info(f"Completed run: {run_name}")
        except Exception as e:
            logger.error(f"Error in run {run_name}: {str(e)}")

if __name__ == "__main__":
    run_sweep()