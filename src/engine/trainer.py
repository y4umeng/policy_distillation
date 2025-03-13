import os
import time
from tqdm import tqdm
import torch
import wandb
from math import ceil
import torch.optim as optim
from collections import OrderedDict
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg
)
from .experience import ReplayBufferDataset
from torch.utils.data import DataLoader

import faulthandler
faulthandler.enable()

class BaseTrainer(object):
    def __init__(self, experiment_name, distiller, env, cfg):
        self.cfg = cfg
        self.env = env
        self.distiller = distiller
        self.replay_buffer = ReplayBufferDataset(cfg.DATA.MAX_CAPACITY)
        self.optimizer = self.init_optimizer(cfg)
        self.num_actions = self.env.action_space.n
        self.best_score = -1

        # init loggers
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.distiller.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
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
        """
        Asynchronous EnvPool version of data collection.
        We'll collect exactly 'num_data_points' teacher-labeled transitions,
        skipping random transitions (like your original code).
        """
        import time
        from tqdm import tqdm
        import torch
        import numpy as np

        self.distiller.teacher.eval()

        # Start all parallel envs in async mode
        self.env.async_reset()

        collected = 0
        time_start = time.time()

        # Optional progress bar
        if self.cfg.LOG.BAR:
            pbar = tqdm(total=num_data_points)

        while collected < num_data_points:
            # 1) Receive a batch of observations, rewards, dones
            obs, rew, terminated, truncated, info = self.env.recv()
            # 'obs' shape is (batch_size, C, H, W) for Atari
            # info["env_id"] shape is (batch_size,)

            env_id = info["env_id"]
            batch_size = obs.shape[0]

            # 2) Decide for each env in the batch: random or teacher
            random_mask = (torch.rand(batch_size) < self.cfg.DATA.EXPLORATION_RATE).cpu().numpy()

            # 3) If not random, do teacher inference on GPU
            obs_tensor = torch.from_numpy(obs).float().to("cuda")  # [batch_size, C, H, W]
            with torch.no_grad():
                policy_dist_all = self.distiller.teacher(obs_tensor)  # [batch_size, num_actions]
            teacher_actions_all = torch.argmax(policy_dist_all, dim=1).cpu().numpy()

            # We'll build the actions array to send back
            actions = np.zeros(batch_size, dtype=np.int64)

            # 4) For each environment in this batch:
            for i in range(batch_size):
                if random_mask[i]:
                    # Pick a random action
                    actions[i] = np.random.randint(self.num_actions)
                    # Skip storing (like your original code with 'continue')
                else:
                    # Pick the teacher's argmax action
                    actions[i] = teacher_actions_all[i]

                    # Immediately store in the replay buffer (like your original code),
                    # which only saves (state, action, policy), not the next state.
                    state_v_int = torch.from_numpy(obs[i]).to(torch.uint8)
                    teacher_action_tensor = torch.tensor(actions[i], dtype=torch.float)
                    policy_dist_tensor = policy_dist_all[i].cpu().float()

                    self.replay_buffer.push(
                        state_v_int,
                        teacher_action_tensor,
                        policy_dist_tensor
                    )

                    collected += 1
                    if self.cfg.LOG.BAR:
                        pbar.update(1)
                        pbar.set_description("Generating data (TRAIN)")

                    # If we've hit the quota of data, we can break out
                    if collected >= num_data_points:
                        break

            # 5) Send the chosen actions back to those envs
            self.env.send(actions, env_id)

            # Break the outer while-loop if we've already collected enough
            if collected >= num_data_points:
                break

        if self.cfg.LOG.BAR:
            pbar.close()

        self.replay_buffer.check_capacity()
        return time.time() - time_start


    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            epoch = state["epoch"] + 1
            self.distiller.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_score = state["best_score"]
            # refill replay buffer
            if "buffer_size" in state and 0:
                self.generate_data(state["buffer_size"])
            else:
                self.generate_data(min(self.cfg.DATA.MAX_CAPACITY, state["epoch"] * self.cfg.DATA.INCREMENT_SIZE))
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            self.train_epoch(epoch)
            epoch += 1
        print(log_msg("Best score:{}".format(self.best_score), "EVAL"))

    def train_epoch(self, epoch):
        lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }

        num_iter = ceil(len(self.replay_buffer)/self.cfg.SOLVER.BATCH_SIZE)

        generatation_time = self.generate_data(self.cfg.DATA.INCREMENT_SIZE)

        data_loader = DataLoader(
            self.replay_buffer,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
        )

        if self.cfg.LOG.BAR: pbar = tqdm(range(num_iter))

        # train loops
        self.distiller.train()
        for idx, data in enumerate(data_loader):
            msg = self.train_iter(data, epoch, train_meters)
            if self.cfg.LOG.BAR:
                pbar.set_description(log_msg(msg, "TRAIN"))
                pbar.update()
        if self.cfg.LOG.BAR: pbar.close()

        # log
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "data_points": len(self.replay_buffer),
                "total_data_gen_time": generatation_time,
                "train_time": train_meters["training_time"].avg,
                "data_load_time": train_meters["data_time"].avg,
            }
        )

        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.distiller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_score": self.best_score,
            "buffer_size": len(self.replay_buffer)
        }
        student_state = {"model": self.distiller.student.state_dict(), "epoch": epoch}
        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(
            student_state, os.path.join(self.log_path, "student_latest")
        )

        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )
            save_checkpoint(
                student_state,
                os.path.join(self.log_path, "student_{}".format(epoch)),
            )

        if epoch == 1 or epoch % self.cfg.LOG.EVAL_FREQ == 0:
            # validate
            eval_start = time.time()
            total_score = validate(self.distiller, self.env, bar=self.cfg.LOG.BAR, num_episodes=self.cfg.LOG.EVAL_EPISODES)
            eval_time = time.time() - eval_start
            log_dict["total_eval_time"] = eval_time
            log_dict["test_score"] = total_score
            # update the best
            if total_score >= self.best_score:
                save_checkpoint(state, os.path.join(self.log_path, "best"))
                save_checkpoint(
                    student_state, os.path.join(self.log_path, "student_best")
                )
                self.best_score = log_dict["test_score"]
                if self.cfg.LOG.WANDB:
                    wandb.run.summary["best_score"] = self.best_score
        
        log_dict["best_score"] = self.best_score
        self.log(lr, epoch, log_dict)

        if not self.cfg.LOG.BAR:
            print(f"Epoch {epoch}. ", end="", flush=True)
            for k, v in log_dict.items():
                print(f"{k}: {v: .3g}, ", end="", flush=True)
            print()

    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, action, teacher_probs = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.cuda(non_blocking=True).to(torch.float32)
        action = action.cuda(non_blocking=True).to(torch.float32)
        teacher_probs = teacher_probs.cuda(non_blocking=True).to(torch.float32)

        # forward
        preds, losses_dict = self.distiller(image=image, target=teacher_probs, epoch=epoch)
        
        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.distiller.get_learnable_parameters(), max_norm=1)
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1 = accuracy(preds, action)
        acc1 = acc1[0]
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
        )
        return msg