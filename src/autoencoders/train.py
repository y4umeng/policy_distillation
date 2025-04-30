import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import wandb
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import ale_py
from yacs.config import CfgNode as CN

from src.autoencoders.models.builder import BuildAutoEncoder
from src.engine.experience import ReplayBufferDataset
from src.engine.utils import (
    AverageMeter, 
    preprocess_env, 
    save_checkpoint, 
    load_checkpoint, 
    log_msg, 
    create_experiment_name
)
from src.engine.cfg import CFG as cfg

cudnn.benchmark = True

class AutoencoderDataset(Dataset):
    """Dataset for autoencoder training using frames from Atari environments"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, state):
        """Add a new state to the buffer"""
        self.buffer.append(state)

    def check_capacity(self):
        if len(self.buffer) > self.capacity:
            self.buffer = self.buffer[-self.capacity:]
            return True
        return False

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        # Return the state at the given index
        return self.buffer[index]

class AutoencoderTrainer:
    def __init__(self, experiment_name, model, env, cfg, resume=False):
        self.cfg = cfg
        self.env = env
        self.model = model
        self.replay_buffer = AutoencoderDataset(cfg.DATA.MAX_CAPACITY)
        self.optimizer = self.init_optimizer(cfg)
        self.criterion = nn.MSELoss()
        self.best_loss = float('inf')

        # init loggers
        self.log_path = self.create_unique_log_path(cfg.LOG.PREFIX, experiment_name, resume)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def create_unique_log_path(self, prefix, experiment_name, resume):
        # Initial log path, if resuming then will override existing files at base_path
        base_path = os.path.join(prefix, experiment_name)
        if resume: return base_path

        log_path = base_path
        counter = 1

        # Check if the path exists and append a number if necessary
        while os.path.exists(log_path):
            log_path = f"{base_path}_{counter}"
            counter += 1

        # Create the directory
        os.makedirs(log_path, exist_ok=True)
        return log_path

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif cfg.SOLVER.TYPE == "Adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=cfg.SOLVER.LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def log(self, lr, epoch, log_dict):
        if self.cfg.LOG.WANDB:
            log_dict["current lr"] = lr
            wandb.log(log_dict)

    def generate_data(self, num_data_points):
        """Generate data by sampling frames from the environment"""
        if not self.cfg.DATA.INCREMENT and len(self.replay_buffer.buffer) == self.replay_buffer.capacity:
            return 0.0

        state, _ = self.env.reset()
        collected = 0
        time_start = time.time()
        if self.cfg.LOG.BAR: 
            pbar = tqdm(range(num_data_points))
            
        while collected < num_data_points:
            # Convert state to proper format
            state_v = torch.tensor(state, dtype=torch.float, requires_grad=False).squeeze()
            state_v_int = state_v.clone().to(torch.uint8)
            
            # Take a random action to explore the environment
            action = self.env.action_space.sample()
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store state in buffer
            self.replay_buffer.push(state_v_int)
            
            state = next_state
            collected += 1

            if done:
                state, _ = self.env.reset()
                
            if self.cfg.LOG.BAR:
                pbar.set_description(log_msg("Generating data", "TRAIN"))
                pbar.update()
                
        if self.cfg.LOG.BAR: 
            pbar.close()
            
        self.replay_buffer.check_capacity()
        return time.time() - time_start

    def train(self, resume=False):
        """Main training loop"""
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_loss = state["best_loss"]
            # Refill replay buffer
            if "buffer_size" in state:
                self.generate_data(state["buffer_size"])
            else:
                self.generate_data(min(self.cfg.DATA.MAX_CAPACITY, state["epoch"] * self.cfg.DATA.INCREMENT_SIZE))
                
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1

        print(log_msg(f"Training completed. Best loss during training: {self.best_loss:.6f}", "EVAL"))
        
        # Load the best model for final evaluation
        print(log_msg("Loading best model for final evaluation...", "EVAL"))
        best_model_path = os.path.join(self.log_path, "best")
        
        try:
            best_model_state = load_checkpoint(best_model_path)
            self.model.load_state_dict(best_model_state["model"])
            
            # Run a final evaluation
            eval_loss = self.evaluate()
            
            print(log_msg(f"Final reconstruction loss: {eval_loss:.6f}", "EVAL"))
            
            if self.cfg.LOG.WANDB:
                wandb.log({
                    "final_evaluation/loss": eval_loss,
                })
                wandb.run.summary["final_loss"] = eval_loss
                
        except Exception as e:
            print(log_msg(f"Error during final evaluation: {str(e)}", "EVAL"))

    def train_epoch(self, epoch):
        """Train for one epoch"""
        # Adjust learning rate based on schedule
        lr = self.adjust_learning_rate(epoch)
        
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
        }

        # Generate new data if needed
        if self.cfg.DATA.INCREMENT:
            generation_time = self.generate_data(self.cfg.DATA.INCREMENT_SIZE)
        else:
            generation_time = self.generate_data(self.cfg.DATA.MAX_CAPACITY)

        # Create data loader
        data_loader = DataLoader(
            self.replay_buffer,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
        )

        num_iter = len(data_loader)
        if self.cfg.LOG.BAR: 
            pbar = tqdm(range(num_iter))

        # Train loop
        self.model.train()
        for idx, state in enumerate(data_loader):
            msg = self.train_iter(state, train_meters)
            if self.cfg.LOG.BAR:
                pbar.set_description(log_msg(msg, "TRAIN"))
                pbar.update()
                
        if self.cfg.LOG.BAR: 
            pbar.close()

        # Run evaluation
        eval_loss = self.evaluate()

        # Log metrics
        log_dict = {
            "train_loss": train_meters["losses"].avg,
            "eval_loss": eval_loss,
            "data_points": len(self.replay_buffer),
            "total_data_gen_time": generation_time,
            "train_time": train_meters["training_time"].avg,
            "data_load_time": train_meters["data_time"].avg,
        }

        # Save checkpoint
        state = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "buffer_size": len(self.replay_buffer)
        }
        
        save_checkpoint(state, os.path.join(self.log_path, "latest"))

        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )

        # Update best model if needed
        if eval_loss < self.best_loss:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            self.best_loss = eval_loss
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_loss"] = self.best_loss
        
        log_dict["best_loss"] = self.best_loss
        self.log(lr, epoch, log_dict)

        if not self.cfg.LOG.BAR:
            print(f"Epoch {epoch}. ", end="", flush=True)
            for k, v in log_dict.items():
                print(f"{k}: {v: .6g}, ", end="", flush=True)
            print()

    def train_iter(self, state, train_meters):
        """Training for a single batch"""
        self.optimizer.zero_grad()
        train_start_time = time.time()
        
        # Move data to GPU
        state = state.to(torch.float32).cuda(non_blocking=True)
        train_meters["data_time"].update(time.time() - train_start_time)
        
        # Forward pass
        output = self.model(state)
        loss = self.criterion(output, state)
        
        # Backward and optimize
        loss.backward()
        self.optimizer.step()
        
        # Update metrics
        train_meters["training_time"].update(time.time() - train_start_time)
        batch_size = state.size(0)
        train_meters["losses"].update(loss.item(), batch_size)
        
        # Return status message
        msg = "Training Epoch | Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.6f}".format(
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
        )
        return msg

    def evaluate(self):
        """Evaluate the model on the replay buffer"""
        self.model.eval()
        eval_loss = 0.0
        eval_count = 0
        
        data_loader = DataLoader(
            self.replay_buffer,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=False,
        )
        
        with torch.no_grad():
            for state in data_loader:
                state = state.to(torch.float32).cuda(non_blocking=True)
                output = self.model(state)
                loss = self.criterion(output, state)
                eval_loss += loss.item() * state.size(0)
                eval_count += state.size(0)
                
        return eval_loss / eval_count if eval_count > 0 else float('inf')

    def adjust_learning_rate(self, epoch):
        """Adjust learning rate based on schedule"""
        steps = np.sum(epoch > np.asarray(self.cfg.SOLVER.LR_DECAY_STAGES))
        if steps > 0:
            new_lr = self.cfg.SOLVER.LR * (self.cfg.SOLVER.LR_DECAY_RATE**steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
            return new_lr
        return self.cfg.SOLVER.LR


def create_autoencoder_config():
    """Create a configuration for autoencoder training"""
    autoencoder_cfg = cfg.clone()
    
    # Update experiment name
    autoencoder_cfg.EXPERIMENT.PROJECT = "autoencoder"
    
    # Update solver settings
    autoencoder_cfg.SOLVER.TYPE = "Adam"  # Adam works better for autoencoders
    autoencoder_cfg.SOLVER.LR = 1e-4
    autoencoder_cfg.SOLVER.EPOCHS = 100
    autoencoder_cfg.SOLVER.BATCH_SIZE = 64
    
    # Update data settings
    autoencoder_cfg.DATA.MAX_CAPACITY = 100000  # Store 100k frames
    autoencoder_cfg.DATA.INCREMENT = True
    autoencoder_cfg.DATA.INCREMENT_SIZE = 10000  # Generate 10k frames per epoch
    
    # Update log settings
    autoencoder_cfg.LOG.PREFIX = "./output/autoencoders"
    autoencoder_cfg.LOG.SAVE_CHECKPOINT_FREQ = 10
    
    return autoencoder_cfg


def main():
    parser = argparse.ArgumentParser(description="Autoencoder training on Atari frames")
    parser.add_argument("--model", type=str, default="vgg16", help="Model architecture (vgg11, vgg13, vgg16, vgg19, resnet18, resnet34, resnet50, etc.)")
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4", help="Atari environment name")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--tag", type=str, default="", help="Tag for the experiment")
    
    args = parser.parse_args()
    
    # Create autoencoder config
    autoencoder_cfg = create_autoencoder_config()
    
    # Update config based on command line arguments
    if args.epochs:
        autoencoder_cfg.SOLVER.EPOCHS = args.epochs
    if args.batch_size:
        autoencoder_cfg.SOLVER.BATCH_SIZE = args.batch_size
    if args.lr:
        autoencoder_cfg.SOLVER.LR = args.lr
    if args.tag:
        autoencoder_cfg.EXPERIMENT.TAG = args.tag
    if args.no_wandb:
        autoencoder_cfg.LOG.WANDB = False
        
    # Set environment
    autoencoder_cfg.DISTILLER.ENV = args.env
    
    # Create experiment name and initialize wandb
    experiment_name = f"autoencoder_{args.model}_{args.env.split('-')[0]}"
    if autoencoder_cfg.LOG.WANDB:
        try:
            wandb.init(
                project=autoencoder_cfg.EXPERIMENT.PROJECT,
                name=experiment_name,
                tags=[autoencoder_cfg.EXPERIMENT.TAG, args.model, args.env],
                config={
                    "model": args.model,
                    "environment": args.env,
                    "batch_size": autoencoder_cfg.SOLVER.BATCH_SIZE,
                    "epochs": autoencoder_cfg.SOLVER.EPOCHS,
                    "learning_rate": autoencoder_cfg.SOLVER.LR,
                    "optimizer": autoencoder_cfg.SOLVER.TYPE,
                }
            )
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            autoencoder_cfg.LOG.WANDB = False
    
    # Initialize environment
    gym.register_envs(ale_py)
    env = preprocess_env(args.env)
    
    # Create model
    model_args = CN()
    model_args.arch = args.model
    model = BuildAutoEncoder(model_args)
    
    # Create trainer
    trainer = AutoencoderTrainer(
        experiment_name,
        model,
        env,
        autoencoder_cfg,
        args.resume
    )
    
    # Start training
    trainer.train(resume=args.resume)
    
    # Close wandb
    if autoencoder_cfg.LOG.WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()