procgen/online/trainer.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import deque
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

import wandb
from online.behavior_policies import PPOnet, ReplayBuffer, algos, arguments, make_venv
from online.evaluation import evaluate
from utils.utils import LogDirType, LogItemType, set_seed


def _save_model(checkpoint: Dict[str, any], directory: str, filename: str) -> None:
    os.makedirs(directory, exist_ok=True)
    saving_path = os.path.expandvars(os.path.expanduser(f"{directory}/{filename}"))
    torch.save(checkpoint, saving_path)


def _load_model(loading_path: str, model: nn.Module, agent) -> int:
    checkpoint = torch.load(loading_path)

    model.load_state_dict(checkpoint[LogItemType.MODEL_STATE_DICT.value])
    agent.optimizer.load_state_dict(checkpoint[LogItemType.OPTIMIZER_STATE_DICT.value])
    curr_epochs = checkpoint[LogItemType.CURRENT_EPOCH.value]
    return curr_epochs


def train(args):
    print("\nArguments: ", args)

    set_seed(args.seed)
    torch.set_num_threads(1)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Create Envs
    envs = make_venv(
        num_envs=args.num_processes,
        env_name=args.env_name,
        device=device,
        **{
            "num_levels": args.num_levels,
            "start_level": args.start_level,
            "distribution_mode": args.distribution_mode,
            "ret_normalization": True,
            "obs_normalization": True,
        },
    )

    # Initialize Model, Agent, and Replay Buffer
    obs_shape = envs.observation_space.shape
    model = PPOnet(obs_shape, envs.action_space.n, base_kwargs={"hidden_size": args.hidden_size})
    model.to(device)
    print("\n Neural Network: ", model)

    agent = algos.PPO(
        model,
        clip_param=args.clip_param,
        ppo_epoch=args.ppo_epoch,
        num_mini_batch=args.num_mini_batch,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
    )
    current_epoch = 0
    if args.resume:
        if os.path.exists(args.checkpoint_path):
            print(f"Trying to load checkpoint from {args.checkpoint_path}")
            current_epoch = _load_model(loading_path=args.checkpoint_path, model=model, agent=agent)
        else:
            loading_path = os.path.expandvars(
                os.path.expanduser(
                    os.path.join(args.model_saving_dir, args.env_name, args.xpid, LogDirType.ROLLING.value, "model.pt")
                )
            )
            if os.path.exists(loading_path):
                print(f"Trying to load checkpoint from {loading_path}")
                current_epoch = _load_model(loading_path=loading_path, model=model, agent=agent)
            else:
                print(
                    f"Loading paths do not exist: {args.checkpoint_path}, {loading_path}! \n"
                    + "Will start training from strach!"
                )

        print(f"Resuming checkpoint from Epoch {current_epoch}")

    rollouts = ReplayBuffer(
        obs_shape,
        action_space=envs.action_space,
        n_envs=args.num_processes,
        n_steps=args.num_steps,
        save_episode=args.save_dataset,
        storage_path=os.path.join(args.dataset_saving_dir, args.env_name, "dataset"),
    )
    obs = envs.reset()
    rollouts.insert_initial_observations(obs)
    rollouts.to(device)

    # Training Loop
    episode_rewards = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    for epoch in range(current_epoch, num_updates):
        model.train()

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = model.act(rollouts._obs_buffer[step])

            obs, reward, done, infos = envs.step(action)

            episode_rewards.extend((info["episode"]["r"] for info in infos if "episode" in info.keys()))

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            rollouts.insert(obs, action, reward, done, action_log_prob, value, masks)

        with torch.no_grad():
            next_value = model.get_value(rollouts._obs_buffer[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        # Parameters are updated in this step
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if epoch % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (epoch + 1) * args.num_processes * args.num_steps
            print("\n")
            print(f"Update {epoch}, step {total_num_steps}:")
            print(
                f"Last {len(episode_rewards)} training episodes, ",
                f"mean/median reward {np.mean(episode_rewards):.2f}/{np.median(episode_rewards):.2f}",
            )

            eval_episode_rewards = evaluate(args=args, model=model, device=device)
            if args.log_wandb:
                wandb.log(
                    {
                        "step": total_num_steps,
                        "current_update_count": epoch,
                        "policy_gradient_loss": action_loss,
                        "value_loss": value_loss,
                        "dist_entropy": dist_entropy,
                        "Train Mean Episode Returns:": np.mean(episode_rewards),
                        "Train Median Episode Returns:": np.median(episode_rewards),
                        "Test Mean Episode Returns:": np.mean(eval_episode_rewards),
                        "Test Median Episode Returns": np.median(eval_episode_rewards),
                    }
                )

            # save model
            if args.model_saving_dir != "":
                checkpoint = {
                    LogItemType.MODEL_STATE_DICT.value: model.state_dict(),
                    LogItemType.OPTIMIZER_STATE_DICT.value: agent.optimizer.state_dict(),
                    LogItemType.CURRENT_EPOCH.value: epoch + 1,
                }

                _save_model(
                    checkpoint=checkpoint,
                    directory=os.path.join(args.model_saving_dir, args.env_name, args.xpid, LogDirType.ROLLING.value),
                    filename="model.pt",
                )

                if epoch % args.archive_interval == 0:
                    _save_model(
                        checkpoint=checkpoint,
                        directory=os.path.join(
                            args.model_saving_dir, args.env_name, args.xpid, LogDirType.CHECKPOINT.value
                        ),
                        filename=f"model_{epoch}_{np.mean(eval_episode_rewards):.2f}.pt",
                    )

    # Save Final Model
    if args.model_saving_dir != "":
        eval_episode_rewards = evaluate(args=args, model=model, device=device)
        _save_model(
            checkpoint={
                LogItemType.MODEL_STATE_DICT.value: model.state_dict(),
                LogItemType.OPTIMIZER_STATE_DICT.value: agent.optimizer.state_dict(),
            },
            directory=os.path.join(args.model_saving_dir, args.env_name, args.xpid, LogDirType.FINAL.value),
            filename=f"model_{np.mean(eval_episode_rewards):.2f}.pt",
        )


def init_wandb(args):
    if (
        args.wandb_base_url is None
        or args.wandb_api_key is None
        or args.wandb_entity is None
        or args.wandb_project is None
    ):
        arguments.parser.error(
            "Either use '--log_wandb=False' or provide WANDB params ! \n"
            + f"BASE_URL: {args.wandb_base_url}, API_KEY: {args.wandb_api_key}, ENTITY: {args.wandb_entity}"
            + f"PROJECT: {args.wandb_project}"
        )

    os.environ["WANDB_BASE_URL"] = args.wandb_base_url
    os.environ["WANDB_API_KEY"] = args.wandb_api_key
    os.environ["WANDB_START_METHOD"] = "thread"
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=args,
        name=args.xpid,
        tags=["vary_n_frames"],
        group=args.xpid[:-2],
    )


if __name__ == "__main__":
    args = arguments.parser.parse_args()

    if args.log_wandb:
        init_wandb(args=args)

    train(args)
```

procgen/offline/train_offline_agent.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import time

import procgen
import torch
import torch.nn as nn
from baselines.common.vec_env import VecExtractDictObs
from torch.utils.data import DataLoader

import wandb
from offline.agents import _create_agent
from offline.arguments import parser
from offline.dataloader import OfflineDataset, OfflineDTDataset
from offline.test_offline_agent import eval_agent, eval_DT_agent
from utils.filewriter import FileWriter
from utils.utils import set_seed
from utils.early_stopper import EarlyStop

args = parser.parse_args()
print(args)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

set_seed(args.seed)

if args.xpid is None:
    args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")

# Setup wandb and offline logging
with open("wandb_info.txt") as file:
    lines = [line.rstrip() for line in file]
    os.environ["WANDB_BASE_URL"] = lines[0]
    os.environ["WANDB_API_KEY"] = lines[1]
    os.environ["WANDB_START_METHOD"] = "thread"
    wandb_group = args.xpid[:-2][:126]  # '-'.join(args.xpid.split('-')[:-2])[:120]
    wandb_project = "OfflineRLBenchmark"
    wandb.init(project=wandb_project, entity=lines[2], config=args, name=args.xpid, group=wandb_group, tags=[args.algo,"final","seed_1","subop","8_june"])

log_dir = os.path.expandvars(os.path.expanduser(os.path.join(args.save_path, args.env_name)))
# check if final_model.pt already exists in the log_dir
if os.path.exists(os.path.join(log_dir, args.xpid, "final_model.pt")):
    # exit if final_model.pt already exists
    print("Final model already exists in the log_dir")
    exit(0)
filewriter = FileWriter(xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir)


def log_stats(stats):
    filewriter.log(stats)
    wandb.log(stats)


# logging.getLogger().setLevel(logging.INFO)

# Load dataset
pin_dataloader_memory = True
extra_config = None
if args.algo in ["dt", "bct"]:
    dataset = OfflineDTDataset(
        capacity=args.dataset_size, episodes_dir_path=os.path.join(args.dataset, args.env_name), percentile=args.percentile, context_len=args.dt_context_length, rtg_noise_prob=args.dt_rtg_noise_prob
    )
    pin_dataloader_memory = True
    extra_config = {"train_data_vocab_size": dataset.vocab_size, "train_data_block_size": dataset._block_size, "max_timesteps": max(dataset._timesteps), "dataset_size": len(dataset)}
    eval_max_return = dataset.get_max_return(multiplier=args.dt_eval_ret)
    print("[DEBUG] Setting max eval return to ", eval_max_return)
else:
    dataset = OfflineDataset(
        capacity=args.dataset_size, episodes_dir_path=os.path.join(args.dataset, args.env_name), percentile=args.percentile
    )
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=pin_dataloader_memory) #, num_workers=8)

print("Dataset Loaded!")

# create Procgen env
env = procgen.ProcgenEnv(num_envs=1, env_name=args.env_name)
env = VecExtractDictObs(env, "rgb")

curr_epochs = 0
last_logged_update_count_at_restart = -1

# Initialize agent
agent = _create_agent(args, env=env, extra_config=extra_config)
agent.set_device(device)
print("Model Created!")

# wandb watch
# wandb.watch(agent.model_actor, log_freq=100)
# wandb.watch(agent.actor_dist, log_freq=100)
# wandb.watch(agent.model_v, log_freq=100)
# wandb.watch(agent.model_q1, log_freq=100)
# wandb.watch(agent.model_q2, log_freq=100)

# load checkpoint and resume if resume flag is true
if args.resume and os.path.exists(os.path.join(args.save_path, args.env_name, args.xpid, "model.pt")):
    curr_epochs = agent.load(os.path.join(args.save_path, args.env_name, args.xpid, "model.pt"))
    last_logged_update_count_at_restart = filewriter.latest_update_count()
    print(f"Resuming checkpoint from Epoch {curr_epochs}, logged update count {last_logged_update_count_at_restart}")  
elif args.resume and os.path.exists(os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt")):
    curr_epochs = agent.load(os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt"))
    last_logged_update_count_at_restart = filewriter.latest_update_count()
    print(f"Resuming checkpoint from Epoch {curr_epochs}, logged update count {last_logged_update_count_at_restart}")
else:
    print("Starting from scratch!")

if args.early_stop:
    early_stopper = EarlyStop(wait_epochs=10, min_delta=0.1)

# Train agent
for epoch in range(curr_epochs, args.epochs):
    agent.train()
    epoch_loss = 0
    epoch_start_time = time.time()
    if args.algo in ["dt", "bct"]:
        for observations, actions, rtgs, timesteps, padding_mask in dataloader:
            observations, actions, rtgs, timesteps, padding_mask = (
                observations.to(device),
                actions.to(device),
                rtgs.to(device),
                timesteps.to(device),
                padding_mask.to(device)
            )
            stats_dict = agent.train_step(observations.float(), actions.long(), rtgs.float(), timesteps.long(), padding_mask.float())
            epoch_loss += stats_dict["loss"]
    else:
        for observations, actions, rewards, next_observations, dones in dataloader:
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(dim=1)
            if len(rewards.shape) == 1:
                rewards = rewards.unsqueeze(dim=1)
            if len(dones.shape) == 1:
                dones = dones.unsqueeze(dim=1)
            observations, actions, rewards, next_observations, dones = (
                observations.to(device),
                actions.to(device),
                rewards.to(device),
                next_observations.to(device),
                dones.to(device),
            )
            stats_dict = agent.train_step(
                observations.float(), actions.long(), rewards.float(), next_observations.float(), dones.float()
            )
            epoch_loss += stats_dict["loss"]
    epoch_end_time = time.time()

    # evaluate the agent on procgen environment
    if epoch % args.eval_freq == 0:
        inf_start_time = time.time()
        if args.algo in ["dt", "bct"]:
            test_mean_perf = eval_DT_agent(
                agent,
                eval_max_return,
                device,
                env_name=args.env_name,
                start_level=args.num_levels+50,
                distribution_mode=args.distribution_mode
            )
            val_mean_perf = eval_DT_agent(
                agent,
                eval_max_return,
                device,
                env_name=args.env_name,
                num_levels=50,
                start_level=args.num_levels,
                distribution_mode=args.distribution_mode
            )
            train_mean_perf = eval_DT_agent(
                agent,
                eval_max_return,
                device,
                env_name=args.env_name,
                num_levels=args.num_levels,
                start_level=0,
                distribution_mode=args.distribution_mode
            )
        else:
            test_mean_perf = eval_agent(
                agent,
                device,
                env_name=args.env_name,
                start_level=args.num_levels+50,
                distribution_mode=args.distribution_mode,
                eval_eps=args.eval_eps,
            )
            val_mean_perf = eval_agent(
                agent,
                device,
                env_name=args.env_name,
                num_levels=50,
                start_level=args.num_levels,
                distribution_mode=args.distribution_mode,
                eval_eps=args.eval_eps,
            )
            train_mean_perf = eval_agent(
                agent,
                device,
                env_name=args.env_name,
                num_levels=args.num_levels,
                start_level=0,
                distribution_mode=args.distribution_mode,
                eval_eps=args.eval_eps,
            )
        inf_end_time = time.time()

        print(
            f"Epoch: {epoch + 1} | Loss: {epoch_loss / len(dataloader)} | Time: {epoch_end_time - epoch_start_time} \
                | Train Returns (mean): {train_mean_perf} | Validation Returns (mean): {val_mean_perf} | Test Returns (mean): {test_mean_perf}"
        )

        print(epoch+1)
        if (epoch+1) > last_logged_update_count_at_restart:
            stats_dict.update(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss / len(dataloader),
                    "epoch_time": epoch_end_time - epoch_start_time,
                    "inf_time": inf_end_time - inf_start_time,
                    "train_rets_mean": train_mean_perf,
                    "test_rets_mean": test_mean_perf,
                    "val_rets_mean": val_mean_perf,
                }
            )
            log_stats(stats_dict)

            # Save agent and number of epochs
            if args.resume and (epoch+1) % args.ckpt_freq == 0:
                curr_epochs = epoch + 1
                agent.save(num_epochs=curr_epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, "model.pt"))
                agent.save(num_epochs=curr_epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, f"model_{epoch}.pt"))
                
        if args.early_stop:
            if early_stopper.should_stop(epoch, val_mean_perf):
                print("[DEBUG]: Early stopping")             
                break

if args.algo in ["dt", "bct"]:
    test_mean_perf = eval_DT_agent(agent, eval_max_return, device, env_name=args.env_name, start_level=args.num_levels+50, distribution_mode=args.distribution_mode)
    train_mean_perf = eval_DT_agent(
        agent, eval_max_return, device, env_name=args.env_name, num_levels=args.num_levels, start_level=0, distribution_mode=args.distribution_mode
    )
    val_mean_perf = eval_DT_agent(
        agent, eval_max_return, device, env_name=args.env_name, num_levels=50, start_level=args.num_levels, distribution_mode=args.distribution_mode
    )
else:
    test_mean_perf = eval_agent(agent, device, env_name=args.env_name, start_level=args.num_levels+50, distribution_mode=args.distribution_mode, eval_eps=args.eval_eps)
    train_mean_perf = eval_agent(
        agent, device, env_name=args.env_name, num_levels=args.num_levels, start_level=0, distribution_mode=args.distribution_mode, eval_eps=args.eval_eps
    )
    val_mean_perf = eval_agent(
        agent, device, env_name=args.env_name, num_levels=50, start_level=args.num_levels, distribution_mode=args.distribution_mode
    )
wandb.log({"final_test_ret": test_mean_perf, "final_train_ret": train_mean_perf, "final_val_ret": val_mean_perf}, step=(epoch + 1))
filewriter.log_final_test_eval({
        'final_test_ret': test_mean_perf,
        'final_train_ret': train_mean_perf,
        'final_val_ret': val_mean_perf
    })
if args.resume:
    agent.save(num_epochs=args.epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt"))
```

procgen/offline/single_level_train_offline_agent.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import time

import numpy as np
import procgen
import torch
import torch.nn as nn
from baselines.common.vec_env import VecExtractDictObs
from torch.utils.data import DataLoader

import wandb
from offline.agents import _create_agent
from offline.arguments import parser
from offline.dataloader import OfflineDataset, OfflineDTDataset
from offline.test_offline_agent import eval_agent, eval_DT_agent
from utils.early_stopper import EarlyStop
from utils.filewriter import FileWriter
from utils.utils import set_seed

args = parser.parse_args()
print(args)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

set_seed(args.seed)

if args.xpid is None:
    args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")

# Setup wandb and offline logging
with open("wandb_info.txt") as file:
    lines = [line.rstrip() for line in file]
    os.environ["WANDB_BASE_URL"] = lines[0]
    os.environ["WANDB_API_KEY"] = lines[1]
    os.environ["WANDB_START_METHOD"] = "thread"
    wandb_group = args.xpid[:-2][:126]  # '-'.join(args.xpid.split('-')[:-2])[:120]
    wandb_project = "OfflineRLBenchmark"
    wandb.init(project=wandb_project, entity=lines[2], config=args, name=args.xpid, group=wandb_group, tags=[args.algo,"single_level"])

log_dir = os.path.expandvars(os.path.expanduser(os.path.join(args.save_path, args.env_name)))
# check if final_model.pt already exists in the log_dir
if os.path.exists(os.path.join(log_dir, args.xpid, "final_model.pt")):
    # exit if final_model.pt already exists
    print("Final model already exists in the log_dir")
    exit(0)
filewriter = FileWriter(xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir)

# episode_info_dict = np.load('./episode_info_dict.npy', allow_pickle=True).item()

# # find the level seed
# max_episodes = None
# max_capacity = args.dataset_size
# if args.capacity_type == "transitions":
#     num_units = {level: sum(episode_info_dict[args.env_name][level]) for level in range(200)}
# elif args.capacity_type == "episodes":
#     num_units = {level: len(episode_info_dict[args.env_name][level]) for level in range(200)}

# if args.threshold_metric=="percentile":
#     percentile_units = np.percentile(list(num_units.values()), 90)
# elif args.threshold_metric=="median":
#     percentile_units = np.median(list(num_units.values()))
# print("Percentile units: ", percentile_units.round())
# all_possible_levels = [level for level in range(200) if num_units[level] >= percentile_units.round()]
# specific_level = int(np.random.choice(all_possible_levels))
# print("Specific level: ", specific_level)
# if args.capacity_type == "episodes":
#     max_episodes = percentile_units.round()
# elif args.capacity_type == "transitions":
#     max_capacity = percentile_units.round()

args.capacity_type = "transitions"
max_capacity = 100000
file_names = os.listdir(os.path.join(args.dataset, args.env_name))[0]
specific_level = int(file_names.split("_")[3])
max_episodes = None

def log_stats(stats):
    filewriter.log(stats)
    wandb.log(stats)


# Load dataset
pin_dataloader_memory = False
extra_config = None
if args.algo in ["dt", "bct"]:
    dataset = OfflineDTDataset(
        capacity=args.dataset_size, episodes_dir_path=os.path.join(args.dataset, args.env_name), percentile=args.percentile, context_len=args.dt_context_length, rtg_noise_prob=args.dt_rtg_noise_prob
    )
    pin_dataloader_memory = True
    extra_config = {"train_data_vocab_size": dataset.vocab_size, "train_data_block_size": dataset._block_size, "max_timesteps": max(dataset._timesteps), "dataset_size": len(dataset)}
    eval_max_return = dataset.get_max_return(multiplier=args.dt_eval_ret)
    print("[DEBUG] Setting max eval return to ", eval_max_return)
else:
    dataset = OfflineDataset(
        capacity=max_capacity, episodes_dir_path=os.path.join(args.dataset, args.env_name), percentile=args.percentile, 
        capacity_type=args.capacity_type, specific_level_seed=specific_level, max_episodes=max_episodes
    )
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=pin_dataloader_memory) #, num_workers=8)

print("Dataset Loaded!")

with open(os.path.join(log_dir, args.xpid, "data_info.csv"), "w") as f:
    f.write("level,episodes,transitions\n")
    f.write(f"{specific_level},{len(dataset._episodes)},{dataset._size}\n")

# create Procgen env
env = procgen.ProcgenEnv(num_envs=1, env_name=args.env_name, start_level=specific_level, num_levels=1, distribution_mode=args.distribution_mode)
env = VecExtractDictObs(env, "rgb")

curr_epochs = 0
last_logged_update_count_at_restart = -1

# Initialize agent
agent = _create_agent(args, env=env, extra_config=extra_config)
agent.set_device(device)
print("Model Created!")

# wandb watch
# wandb.watch(agent.model_actor, log_freq=100)
# wandb.watch(agent.actor_dist, log_freq=100)
# wandb.watch(agent.model_v, log_freq=100)
# wandb.watch(agent.model_q1, log_freq=100)
# wandb.watch(agent.model_q2, log_freq=100)

# load checkpoint and resume if resume flag is true
if args.resume and os.path.exists(os.path.join(args.save_path, args.env_name, args.xpid, "model.pt")):
    curr_epochs = agent.load(os.path.join(args.save_path, args.env_name, args.xpid, "model.pt"))
    last_logged_update_count_at_restart = filewriter.latest_update_count()
    print(f"Resuming checkpoint from Epoch {curr_epochs}, logged update count {last_logged_update_count_at_restart}")  
elif args.resume and os.path.exists(os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt")):
    curr_epochs = agent.load(os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt"))
    last_logged_update_count_at_restart = filewriter.latest_update_count()
    print(f"Resuming checkpoint from Epoch {curr_epochs}, logged update count {last_logged_update_count_at_restart}")
else:
    print("Starting from scratch!")

if args.early_stop:
    early_stopper = EarlyStop(wait_epochs=10, min_delta=0.1)

# Train agent
for epoch in range(curr_epochs, args.epochs):
    agent.train()
    epoch_loss = 0
    epoch_start_time = time.time()
    if args.algo in ["dt", "bct"]:
        for observations, actions, rtgs, timesteps, padding_mask in dataloader:
            observations, actions, rtgs, timesteps, padding_mask = (
                observations.to(device),
                actions.to(device),
                rtgs.to(device),
                timesteps.to(device),
                padding_mask.to(device)
            )
            stats_dict = agent.train_step(observations.float(), actions.long(), rtgs.float(), timesteps.long(), padding_mask.float())
            epoch_loss += stats_dict["loss"]
    else:
        for observations, actions, rewards, next_observations, dones in dataloader:
            if len(actions.shape) == 1:
                actions = actions.unsqueeze(dim=1)
            if len(rewards.shape) == 1:
                rewards = rewards.unsqueeze(dim=1)
            if len(dones.shape) == 1:
                dones = dones.unsqueeze(dim=1)
            observations, actions, rewards, next_observations, dones = (
                observations.to(device),
                actions.to(device),
                rewards.to(device),
                next_observations.to(device),
                dones.to(device),
            )
            stats_dict = agent.train_step(
                observations.float(), actions.long(), rewards.float(), next_observations.float(), dones.float()
            )
            epoch_loss += stats_dict["loss"]
    epoch_end_time = time.time()

    # evaluate the agent on procgen environment
    if epoch % args.eval_freq == 0:
        inf_start_time = time.time()
        if args.algo in ["dt", "bct"]:
            test_mean_perf = eval_DT_agent(
                agent,
                eval_max_return,
                device,
                env_name=args.env_name,
                start_level=specific_level,
                num_levels=1,
                distribution_mode=args.distribution_mode
            )
        else:
            test_mean_perf = eval_agent(
                agent,
                device,
                env_name=args.env_name,
                start_level=specific_level,
                num_levels=1,
                distribution_mode=args.distribution_mode,
                eval_eps=args.eval_eps,
            )
        inf_end_time = time.time()

        print(
            f"Epoch: {epoch + 1} | Loss: {epoch_loss / len(dataloader)} | Time: {epoch_end_time - epoch_start_time} \
             | Test Returns (mean): {test_mean_perf}"
        )

        print(epoch+1)
        if (epoch+1) > last_logged_update_count_at_restart:
            stats_dict.update(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss / len(dataloader),
                    "epoch_time": epoch_end_time - epoch_start_time,
                    "inf_time": inf_end_time - inf_start_time,
                    "test_rets_mean": test_mean_perf,
                }
            )
            log_stats(stats_dict)

            # Save agent and number of epochs
            if args.resume and (epoch+1) % args.ckpt_freq == 0:
                curr_epochs = epoch + 1
                agent.save(num_epochs=curr_epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, "model.pt"))
                agent.save(num_epochs=curr_epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, f"model_{epoch}.pt"))
                
        if args.early_stop:
            if early_stopper.should_stop(epoch, test_mean_perf):
                print("[DEBUG]: Early stopping")             
                break

if args.algo in ["dt", "bct"]:
    test_mean_perf = eval_DT_agent(agent, eval_max_return, device, env_name=args.env_name, start_level=specific_level, num_levels=1, distribution_mode=args.distribution_mode, num_episodes=100)
else:
    test_mean_perf = eval_agent(agent, device, env_name=args.env_name, start_level=specific_level, num_levels=1, distribution_mode=args.distribution_mode, eval_eps=args.eval_eps, num_episodes=100)

print("Final Test Returns (mean): ", test_mean_perf)
# save dict to csv in logdir
with open(os.path.join(log_dir, args.xpid, "evaluate.csv"), "w") as f:
    f.write("final_test_ret,final_train_ret,final_val_ret\n")
    f.write(f"{test_mean_perf},{test_mean_perf},{test_mean_perf}\n")

if args.resume:
    agent.save(num_epochs=args.epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt"))
```

procgen/offline/agents/__init__.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from offline.agents.bc import BehavioralCloning
from offline.agents.bcq import BCQ
from offline.agents.ddqn_cql import CQL
from offline.agents.iql import IQL
from offline.agents.dt import DecisionTransformer

def _create_agent(args, env, extra_config):
    agent_name = args.algo
    if agent_name == "bc":
        return BehavioralCloning(env.observation_space, env.action_space.n, args.lr, args.agent_model, args.hidden_size)
    if agent_name == "bcq":
        assert args.agent_model in ["bcq", "bcqresnetbase"]
        return BCQ(env.observation_space, 
                   env.action_space.n, 
                   args.lr, 
                   args.agent_model, 
                   args.hidden_size,
                   gamma=args.gamma,
                   target_update_freq=args.target_update_freq,
                   tau=args.tau,
                   eps_start=args.eps_start,
                   eps_end=args.eps_end,
                   eps_decay=args.eps_decay,
                   bcq_threshold=args.bcq_threshold,
                   perform_polyak_update=args.perform_polyak_update)
    elif agent_name == "cql":
        return CQL(env.observation_space, 
                   env.action_space.n, 
                   args.lr, 
                   args.agent_model, 
                   args.hidden_size,
                   gamma=args.gamma,
                   target_update_freq=args.target_update_freq,
                   tau=args.tau,
                   eps_start=args.eps_start,
                   eps_end=args.eps_end,
                   eps_decay=args.eps_decay,
                   cql_alpha=args.cql_alpha,
                   perform_polyak_update=args.perform_polyak_update)
    elif agent_name == "iql":
        return IQL(env.observation_space, 
                   env.action_space.n, 
                   args.lr, 
                   args.agent_model, 
                   args.hidden_size,
                   gamma=args.gamma,
                   target_update_freq=args.target_update_freq,
                   tau=args.tau,
                   eps_start=args.eps_start,
                   eps_end=args.eps_end,
                   eps_decay=args.eps_decay,
                   iql_temperature=args.iql_temperature,
                   iql_expectile=args.iql_expectile,
                   perform_polyak_update=args.perform_polyak_update)
    elif agent_name in ["dt", "bct"]:
        return DecisionTransformer(env.observation_space,
                                   env.action_space.n, 
                                    args.agent_model, 
                                    extra_config["train_data_vocab_size"],
                                    extra_config["train_data_block_size"],
                                    extra_config["max_timesteps"],
                                    args.dt_context_length,
                                    extra_config["dataset_size"],
                                    lr=args.lr)
    else:
        raise ValueError(f"Invalid agent name {agent_name}.")
```

procgen/offline/agents/bc.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import AGENT_CLASSES
from online.behavior_policies.distributions import Categorical


class BehavioralCloning:
    def __init__(self, observation_space, action_space, lr, agent_model, hidden_size=64):
        """
        Initialize the agent.

        :param observation_space: the observation space for the environment
        :param action_space: the action space for the environment
        :param lr: the learning rate for the agent
        :param hidden_size: the size of the hidden layers for the agent
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.hidden_size = hidden_size

        self.model_base = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size, use_actor_linear=False)
        self.model_dist = Categorical(hidden_size, self.action_space)
        self.optimizer = torch.optim.Adam(list(self.model_base.parameters()) + list(self.model_dist.parameters()), lr=self.lr)
        
        self.total_steps = 0

    def train(self):
        self.model_base.train()
        self.model_dist.train()

    def eval(self):
        self.model_base.eval()
        self.model_dist.eval()

    def set_device(self, device):
        self.model_base.to(device)
        self.model_dist.to(device)

    def eval_step(self, observation, eps=0.0):
        """
        Given an observation, return an action.

        :param observation: the observation for the environment
        :return: the action for the environment in numpy
        """
        deterministic = eps == 0.0
        with torch.no_grad():
            actor_features = self.model_base(observation)
            dist = self.model_dist(actor_features)
            
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

        return action.cpu().numpy()

    def train_step(self, observations, actions, rewards, next_observations, dones):
        """
        Update the agent given observations and actions.

        :param observations: the observations for the environment
        :param actions: the actions for the environment
        """
        # squeeze actions to [batch_size] if they are [batch_size, 1]
        if len(actions.shape) == 2:
            actions = actions.squeeze(dim=1)
            
        actor_features = self.model_base(observations)
        dist = self.model_dist(actor_features)
        action_log_probs = dist._get_log_softmax()
        
        self.optimizer.zero_grad()
        loss = F.nll_loss(action_log_probs, actions)
        loss.backward()
        self.optimizer.step()
        self.total_steps += 1
        # create stats dict
        stats = {"loss": loss.item(), "total_steps": self.total_steps}
        return stats

    def save(self, num_epochs, path):
        """
        Save the model to a given path.

        :param path: the path to save the model
        """
        save_dict = {
            "model_base_state_dict": self.model_base.state_dict(),
            "model_dist_state_dict": self.model_dist.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "curr_epochs": num_epochs
        }
        torch.save(save_dict, path)
        return

    def load(self, path):
        """
        Load the model from a given path.

        :param path: the path to load the model
        """
        checkpoint = torch.load(path)
        self.model_base.load_state_dict(checkpoint["model_base_state_dict"])
        self.model_dist.load_state_dict(checkpoint["model_dist_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        return checkpoint["curr_epochs"]
```

procgen/offline/agents/bcq.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as  F
import numpy as np
import math

from utils import AGENT_CLASSES

class BCQ:
	def __init__(self, 
				 observation_space, 
				 action_space, 
				 lr, 
				 agent_model, 
				 hidden_size, 
				 gamma, 
				 target_update_freq, 
				 tau,
				 eps_start, 
				 eps_end, 
				 eps_decay, 
				 bcq_threshold,
				 perform_polyak_update):
		"""
		Initialize the agent.

		:param observation_space: the observation space for the environment
		:param action_space: the action space for the environment
		:param lr: the learning rate for the agent
		:param hidden_size: the size of the hidden layers for the agent
		:param gamma: the discount factor for the agent
		:param target_update_freq: the frequency with which to update the target network
		:param tau: the soft update factor for the target network
		:param eps_start: the starting epsilon for the agent
		:param eps_end: the ending epsilon for the agent
		:param eps_decay: the decay rate for epsilon
		:param bcq_threshold: the threshold for selecting the best action
		:param perform_polyak_update: whether to use polyak averaging or not
		"""
		self.observation_space = observation_space
		self.action_space = action_space
		self.lr = lr
		self.hidden_size = hidden_size
		self.gamma = gamma
		self.target_update_freq = target_update_freq
		self.tau = tau
		
		self.model = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		self.target_model = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		
		self.target_model.load_state_dict(self.model.state_dict())
		self.target_model.eval()
		
		self.total_steps = 0
		
		self.eps_start = eps_start
		self.eps_end = eps_end
		self.eps_decay = eps_decay
		self.bcq_threshold = bcq_threshold
		self.perform_polyak_update = perform_polyak_update
		
	def train(self):
		self.model.train()

	def eval(self):
		self.model.eval()
		
	def set_device(self, device):
		self.model.to(device)
		self.target_model.to(device)
		
	def train_step(self, observations, actions, rewards, next_observations, dones):
		with torch.no_grad():
			# Q-values for best actions in next observations
			next_q_values_model, next_action_probs, _ = self.model(next_observations) # [batch_size, num_actions]
			next_action_probs = next_action_probs.exp()
			next_action_probs = (next_action_probs/next_action_probs.max(1, keepdim=True)[0] > self.bcq_threshold).float()
			# Use large negative number to mask actions from argmax
			next_actions = (next_action_probs * next_q_values_model + (1 - next_action_probs) * -1e8).argmax(1, keepdim=True) # [batch_size, 1]
			next_q_values, _, _ = self.target_model(next_observations)
			next_q_value = next_q_values.gather(1, next_actions).reshape(-1, 1)
			# Compute the target of the current Q-values
			target_q_values = rewards + (1 - dones) * self.gamma * next_q_value # [batch_size, 1]
			
		# Q-values for current observations
		q_values, curr_action_probs, curr_action_i = self.model(observations)
		# Compute the predicted q values for the actions taken
		pred_q_values = q_values.gather(1, actions)
		
		# Train the model with Bellman error as targets
		ddqn_loss = F.smooth_l1_loss(pred_q_values, target_q_values)
		
		# Calculate BCQ loss
		i_loss = F.nll_loss(curr_action_probs, actions.reshape(-1))
		bcq_loss = i_loss + 1e-2 * curr_action_i.pow(2).mean()
		
		loss = ddqn_loss + bcq_loss
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		# Update the target network, copying all weights and biases in DQN
		if self.perform_polyak_update:
			self.soft_update_target(tau=self.tau)
		else:
			if self.total_steps % self.target_update_freq == 0:
				self.copy_update_target()
		
		self.total_steps += 1
		
		# create stats dict
		stats = {"loss": loss.item(), "ddqn_loss": ddqn_loss.item(), "bcq_loss": bcq_loss.item(), "total_steps": self.total_steps}
		return stats
	
	def soft_update_target(self, tau):
		"""
		Perform a soft update of the target network.
		:param tau: the soft update coefficient
		"""
		for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
			target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
			
	def copy_update_target(self):
		"""
		Perform a duplicate update of the target network.
		"""
		self.target_model.load_state_dict(self.model.state_dict())
			
	@property
	def calculate_eps(self):
		"""
		Calculate epsilon given the current timestep, initial epsilon, end epsilon and decay rate, where initial_eps > end_eps.
		"""
		
		eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.total_steps / self.eps_decay)
		return eps
	
	def eval_step(self, observations, eps=0.001):
		"""
		Given an observation, return an action.

		:param observation: the observation for the environment
		:param eps: the epsilon value for epsilon-greedy action selection
		:return: the action for the environment in numpy
		"""
		self.model.eval()
		# Epsilon-greedy action selection
		if np.random.uniform(0,1) <= eps:
			action = np.random.randint(self.action_space, size=(1,))
		else:
			with torch.no_grad():
				q_values, action_probs, _ = self.model(observations)
				action_probs = action_probs.exp()
				action_probs = (action_probs/action_probs.max(1, keepdim=True)[0] > self.bcq_threshold).float()
				# Use large negative number to mask actions from argmax
				action = (action_probs * q_values + (1. - action_probs) * -1e8).argmax(1).detach().cpu().numpy()
		return action
	
	def save(self, num_epochs, path):
		"""
		Save the model to a given path.

		:param path: the path to save the model
		"""
		save_dict = {
			"model_state_dict": self.model.state_dict(),
			"target_model_state_dict": self.target_model.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
			"total_steps": self.total_steps,
			"curr_epochs": num_epochs
		}
		torch.save(save_dict, path)
		return

	def load(self, path):
		"""
		Load the model from a given path.

		:param path: the path to load the model
		"""
		checkpoint = torch.load(path)
		self.model.load_state_dict(checkpoint["model_state_dict"])
		self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
		self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		self.total_steps = checkpoint["total_steps"]
		self.target_model.eval()
		return checkpoint["curr_epochs"]
			
```

procgen/offline/agents/ddqn_cql.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as  F
import numpy as np
import math

from utils import AGENT_CLASSES

class CQL:
	def __init__(self, 
				 observation_space, 
				 action_space, 
				 lr, 
				 agent_model, 
				 hidden_size,
				 gamma, 
				 target_update_freq, 
				 tau,
				 eps_start, 
				 eps_end, 
				 eps_decay, 
				 cql_alpha,
     			 perform_polyak_update):
		"""
		Initialize the agent.

		:param observation_space: the observation space for the environment
		:param action_space: the action space for the environment
		:param lr: the learning rate for the agent
		:param hidden_size: the size of the hidden layers for the agent
		:param gamma: the discount factor for the agent
		:param target_update_freq: the frequency with which to update the target network
		:param tau: the soft update factor for the target network
		:param eps_start: the starting epsilon for the agent
		:param eps_end: the ending epsilon for the agent
		:param eps_decay: the decay rate for epsilon
		:param cq_alpha: the alpha value for CQL
		:param perform_polyak_update: whether to use polyak averaging or not
		"""
		self.observation_space = observation_space
		self.action_space = action_space
		self.lr = lr
		self.hidden_size = hidden_size
		self.gamma = gamma
		self.target_update_freq = target_update_freq
		self.tau = tau
		
		self.model = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		self.target_model = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		
		self.target_model.load_state_dict(self.model.state_dict())
		self.target_model.eval()
		
		self.total_steps = 0
		
		self.eps_start = eps_start
		self.eps_end = eps_end
		self.eps_decay = eps_decay
		self.cql_alpha = cql_alpha
		self.perform_polyak_update = perform_polyak_update
		
	def train(self):
		self.model.train()

	def eval(self):
		self.model.eval()
		
	def set_device(self, device):
		self.model.to(device)
		self.target_model.to(device)
		
	def train_step(self, observations, actions, rewards, next_observations, dones):
		# Q-values for current observations
		q_values = self.model(observations) # [batch_size, num_actions]
		with torch.no_grad():
			# Q-values for best actions in next observations
			next_q_values = self.target_model(next_observations)
			next_actions = torch.argmax(self.model(next_observations), dim=1).unsqueeze(1) # [batch_size, 1]
			next_q_value = next_q_values.gather(1, next_actions) # [batch_size, 1]
			# Compute the target of the current Q-values
			target_q_values = rewards + (1 - dones) * self.gamma * next_q_value
		# Compute the predicted q values for the actions taken
		pred_q_values = q_values.gather(1, actions)
		
		# Train the model with Bellman error as targets
		ddqn_loss = F.smooth_l1_loss(pred_q_values, target_q_values)
		
		# Calculate CQL loss
		logsumexp_q_values = torch.logsumexp(q_values, dim=1, keepdim=True) # [batch_size, 1]
		one_hot_actions = F.one_hot(actions.squeeze(dim=1), self.action_space) # [batch_size, num_actions]
		q_values_selected = torch.sum(q_values * one_hot_actions, dim=1, keepdim=True) # [batch_size, 1]
		cql_loss = self.cql_alpha * torch.mean(logsumexp_q_values - q_values_selected)
		
		loss = ddqn_loss + cql_loss
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		# Update the target network, copying all weights and biases in DQN
		if self.perform_polyak_update:
			self.soft_update_target(tau=self.tau)
		else:
			if self.total_steps % self.target_update_freq == 0:
				self.copy_update_target()
		
		self.total_steps += 1
		
		# create stats dict
		stats = {"loss": loss.item(), "ddqn_loss": ddqn_loss.item(), "cql_loss": cql_loss.item(), "eps": self.calculate_eps, "total_steps": self.total_steps}
		return stats
	
	def soft_update_target(self, tau):
		"""
		Perform a soft update of the target network.

		:param tau: the soft update coefficient
		"""
		for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
			target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
			
	def copy_update_target(self):
		"""
		Perform a duplicate update of the target network.
		"""
		self.target_model.load_state_dict(self.model.state_dict())
			
	@property
	def calculate_eps(self):
		"""
		Calculate epsilon given the current timestep, initial epsilon, end epsilon and decay rate.
		"""
		eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.total_steps / self.eps_decay)
		return eps
	
	def eval_step(self, observations, eps=0.5):
		"""
		Given an observation, return an action.

		:param observation: the observation for the environment
		:param eps: the epsilon value for epsilon-greedy action selection
		:return: the action for the environment in numpy
		"""
		self.model.eval()
		# Epsilon-greedy action selection
		if np.random.uniform(0,1) < eps:
			action = np.random.randint(self.action_space, size=(1,))
		else:
			with torch.no_grad():
				q_values = self.model(observations)
				action = torch.argmax(q_values, dim=1).detach().cpu().numpy()
		return action
	
	def save(self, num_epochs, path):
		"""
		Save the model to a given path.

		:param path: the path to save the model
		"""
		save_dict = {
			"model_state_dict": self.model.state_dict(),
			"target_model_state_dict": self.target_model.state_dict(),
			"optimizer_state_dict": self.optimizer.state_dict(),
			"total_steps": self.total_steps,
			"curr_epochs": num_epochs
		}
		torch.save(save_dict, path)
		return

	def load(self, path):
		"""
		Load the model from a given path.

		:param path: the path to load the model
		"""
		checkpoint = torch.load(path)
		self.model.load_state_dict(checkpoint["model_state_dict"])
		self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
		self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
		self.total_steps = checkpoint["total_steps"]
		self.target_model.eval()
		return checkpoint["curr_epochs"]
			
```

procgen/offline/agents/dt.py
```python
"""
This file contains training loop implementing Decision Transformer.

Source:
1. https://github.com/kzl/decision-transformer/blob/master/atari/mingpt
2. https://github.com/karpathy/minGPT
3. https://github.com/karpathy/nanoGPT

----------------------------------------------------------------------------------------------------------------------------

MIT License

Copyright (c) 2021 Decision Transformer (Decision Transformer: Reinforcement Learning via Sequence Modeling) Authors (https://arxiv.org/abs/2106.01345)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

-------------

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as  F
from torch.distributions import Categorical
from utils.gpt_arch import GPT, GPTConfig
import numpy as np
import math


class DTConfig:
    # optimization parameters
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
            
class DecisionTransformer:
    def __init__(self, 
                 observation_space, 
                 action_space,  
                 agent_model, 
                 train_data_vocab_size,
                 train_data_block_size,
                 max_timesteps,
                 context_len,
                 dataset_size,
                 lr=6e-4,
                 betas=(0.9, 0.95),
                 grad_norm_clip=1.0,
                 weight_decay=0.1,
                 lr_decay=True,
                 warmup_tokens=512*20):
        """
        Initialize the agent.

        :param observation_space: the observation space for the environment
        :param action_space: the action space for the environment
        :param lr: the learning rate for the agent
        :param agent_model: the agent model type: [reward_conditioned (DT), naive (BCT)]
        :param train_data_vocab_size: the size of the vocabulary for the agent
        :param train_data_block_size: the block size for the agent
        :param max_timesteps: the number of timesteps for the agent
        :param context_len: the context length for the agent
        :param dataset_size: the dataset size for the agent
        :param betas: the betas for the agent
        :param grad_norm_clip: the gradient norm clip for the agent
        :param weight_decay: the weight decay only applied on matmul weights
        :param lr_decay: the learning rate decay with linear warmup followed by cosine decay to 10% of original
        :param warmup_tokens: the warmup tokens
        :param final_tokens: the final tokens (at what point we reach 10% of original LR)
        """
        
        # Initialise GPT config and GPT model
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.agent_model = agent_model
        
        self.train_data_vocab_size = train_data_vocab_size
        self.train_data_block_size = train_data_block_size
        self.model_type = "reward_conditioned" if agent_model == "dt_reward_conditioned" else "naive"
        self.max_timesteps = max_timesteps
        self.final_tokens=2*dataset_size*context_len*3
        
        self.betas = betas
        self.grad_norm_clip = grad_norm_clip
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.warmup_tokens = warmup_tokens
        
        self.mconf = GPTConfig(self.train_data_vocab_size, self.train_data_block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=self.model_type, max_timestep=self.max_timesteps, inp_channels=3)
        self.model = GPT(self.mconf)
        
        self.config = DTConfig(learning_rate=self.lr, lr_decay=self.lr_decay, warmup_tokens=self.warmup_tokens, final_tokens=self.final_tokens,
                      num_workers=4, model_type=self.model_type, max_timestep=self.max_timesteps)
        
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = self.raw_model.configure_optimizers(self.config)
        
        self.total_steps = 0
        self.tokens = 0
        
    def set_device(self, device):
        self.model.to(device)
        
    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()
        
    def top_k_logits(self, logits, k):
        # Source: https://github.com/kzl/decision-transformer/blob/master/atari/mingpt/utils.py
        v, _ = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out
    
    def entropy(self, logits, probs):
        '''
        Calculates mean entropy
        Source: https://pytorch.org/docs/stable/_modules/torch/distributions/categorical.html#Categorical.entropy
        '''
        min_real = torch.finfo(logits.dtype).min
        logits = torch.clamp(logits, min=min_real)
        p_log_p = logits * probs
        return -p_log_p.sum(-1)

    def get_categorical_entropy(self, prob):
        # print(prob.shape)
        distb = Categorical(torch.tensor(prob))
        return distb.entropy()

    @torch.no_grad()
    def sample(self, x, steps, temperature=1.0, sample=False, top_k=None, actions=None, rtgs=None, timesteps=None, return_probs=False):
        """
        take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
        the sequence, feeding the predictions back into the model each time. Clearly the sampling
        has quadratic complexity unlike an RNN that is only linear, and has a finite context window
        of block_size, unlike an RNN that has an infinite context window.
        
        Source: https://github.com/kzl/decision-transformer/blob/master/atari/mingpt/utils.py
        """
        block_size = self.model.get_block_size()
        self.model.eval()
        for k in range(steps):
            # x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
            x_cond = x if x.size(1) <= block_size//3 else x[:, -block_size//3:] # crop context if needed
            if actions is not None:
                actions = actions if actions.size(1) <= block_size//3 else actions[:, -block_size//3:] # crop context if needed
            rtgs = rtgs if rtgs.size(1) <= block_size//3 else rtgs[:, -block_size//3:] # crop context if needed
            logits, _ = self.model(x_cond, actions=actions, targets=None, rtgs=rtgs, timesteps=timesteps)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            # x = torch.cat((x, ix), dim=1)
            x = ix

        if return_probs:
            bc_entropy = self.get_categorical_entropy(probs)
            return x, probs, bc_entropy
        return x
    
    def train_step(self, observations, actions, rtgs, timesteps, padding_mask):

        _, loss = self.model(observations, actions, actions, rtgs, timesteps, padding_mask)
        loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
        
        self.model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()
        
        # decay the learning rate based on our progress
        if self.config.lr_decay:
            self.tokens += (actions >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
            if self.tokens < self.config.warmup_tokens:
                # linear warmup
                lr_mult = float(self.tokens) / float(max(1, self.config.warmup_tokens))
            else:
                # cosine learning rate decay
                progress = float(self.tokens - self.config.warmup_tokens) / float(max(1, self.config.final_tokens - self.config.warmup_tokens))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = self.config.learning_rate * lr_mult
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = self.config.learning_rate
        
        self.total_steps += 1
        
        # create stats dict
        stats = {"loss": loss.item(), "lr": lr, "total_steps": self.total_steps}
        return stats
    
    def save(self, num_epochs, path):
        """
        Save the model to a given path.

        :param path: the path to save the model
        """
        save_dict = {
            "model_save_dict": self.model.state_dict(),
            "tokens_processed": self.tokens,
            "total_steps": self.total_steps,
            "curr_epochs": num_epochs
        }
        torch.save(save_dict, path)
        return
    
    def load(self, path):
        """
        Load the model from a given path.

        :param path: the path to load the model
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_save_dict"])
        self.tokens = checkpoint["tokens_processed"]
        self.total_steps = checkpoint["total_steps"]
        
        return checkpoint["curr_epochs"]
```

procgen/offline/agents/iql.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from online.behavior_policies.distributions import Categorical
from utils import AGENT_CLASSES


class IQL:
	def __init__(
		self,
		observation_space,
		action_space,
		lr,
		agent_model,
		hidden_size,
		gamma,
		target_update_freq,
		tau,
		eps_start,
		eps_end,
		eps_decay,
		iql_temperature,
		iql_expectile,
		perform_polyak_update
	):
		"""
		Initialize the agent.

		:param observation_space: the observation space for the environment
		:param action_space: the action space for the environment
		:param lr: the learning rate for the agent
		:param hidden_size: the size of the hidden layers for the agent
		:param gamma: the discount factor for the agent
		:param target_update_freq: the frequency with which to update the target network
		:param tau: the soft update factor for the target network
		:param eps_start: the starting epsilon for the agent
		:param eps_end: the ending epsilon for the agent
		:param eps_decay: the decay rate for epsilon
		:param iql_temperature: the temperature for the IQL agent
		:param iql_expectile: the expectile weight for the IQL agent
		"""

		# Implement Implicit Q Learning, which has an Actor, Critic and Q Function
		self.observation_space = observation_space
		self.action_space = action_space
		self.lr = lr
		self.hidden_size = hidden_size
		self.gamma = gamma
		self.target_update_freq = target_update_freq
		self.tau = tau

		self.total_steps = 0

		self.eps_start = eps_start
		self.eps_end = eps_end
		self.eps_decay = eps_decay

		self.iql_temperature = iql_temperature
		self.iql_expectile = iql_expectile
		
		self.perform_polyak_update = perform_polyak_update

		# Implement the Actor, Critic and Value Function
		self.model_actor = AGENT_CLASSES[agent_model](
			observation_space, action_space, hidden_size, use_actor_linear=False
		)
		self.actor_dist = Categorical(hidden_size, self.action_space)
		# optimizer_actor uses parameters from model_actor and actor_dist
		actor_model_params = list(self.model_actor.parameters()) + list(self.actor_dist.parameters())
		self.optimizer_actor = torch.optim.Adam(actor_model_params, lr=self.lr)

		self.model_v = AGENT_CLASSES[agent_model](observation_space, 1, hidden_size)
		self.optimizer_v = torch.optim.Adam(self.model_v.parameters(), lr=self.lr)

		self.model_q1 = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		self.optimizer_q1 = torch.optim.Adam(self.model_q1.parameters(), lr=self.lr)
		self.target_q1 = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		self.target_q1.load_state_dict(self.model_q1.state_dict())
		self.target_q1.eval()

		self.model_q2 = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		self.optimizer_q2 = torch.optim.Adam(self.model_q2.parameters(), lr=self.lr)
		self.target_q2 = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size)
		self.target_q2.load_state_dict(self.model_q2.state_dict())
		self.target_q2.eval()

	def train(self):
		self.model_actor.train()
		self.actor_dist.train()
		self.model_v.train()
		self.model_q1.train()
		self.model_q2.train()

	def eval(self):
		self.model_actor.eval()
		self.actor_dist.eval()
		self.model_v.eval()
		self.model_q1.eval()
		self.model_q2.eval()

	def set_device(self, device):
		self.model_actor.to(device)
		self.actor_dist.to(device)
		self.model_v.to(device)
		self.model_q1.to(device)
		self.model_q2.to(device)
		self.target_q1.to(device)
		self.target_q2.to(device)

	def expectile_loss(self, u_diff, expectile=0.8):
		"""
		Calculate the expectile loss for the IQL agent.

		:param value: the value function shape [batch_size, 1]
		:param Q_value: the Q-value shape [batch_size, 1]
		:param expectile: the expectile weight
		"""
		# expectile_weight = torch.where(u_diff > 0, expectile, 1 - expectile)  # [batch_size, 1]
		# L2_tau = expectile_weight * (u_diff**2)  # [batch_size, 1]
		return torch.mean(torch.abs(expectile - (u_diff < 0).float()) * u_diff**2)

	def train_step(self, observations, actions, rewards, next_observations, dones):
		# 1. Calculate Value Loss
		with torch.no_grad():
			q1 = self.target_q1(observations).gather(1, actions)  # [batch_size, 1]
			q2 = self.target_q2(observations).gather(1, actions)  # [batch_size, 1]
			q_minimum = torch.min(q1, q2)  # [batch_size, 1]

		curr_value = self.model_v(observations)  # [batch_size, 1]
		u_diff = q_minimum - curr_value  # [batch_size, 1]
		value_loss = self.expectile_loss(u_diff, self.iql_expectile)  # [1]
		self.optimizer_v.zero_grad(set_to_none=True)
		value_loss.backward()
		self.optimizer_v.step()

		# 2. Calculate Critic Loss
		with torch.no_grad():
			next_v = self.model_v(next_observations)  # [batch_size, 1]
		target_q = rewards + (1 - dones) * self.gamma * next_v.detach()  # [batch_size, 1]
		curr_q1 = self.model_q1(observations).gather(1, actions)  # [batch_size, 1]
		curr_q2 = self.model_q2(observations).gather(1, actions)  # [batch_size, 1]
		critic1_loss = F.mse_loss(curr_q1, target_q).mean()  # [1]

		self.optimizer_q1.zero_grad(set_to_none=True)
		critic1_loss.backward()
		self.optimizer_q1.step()

		critic2_loss = F.mse_loss(curr_q2, target_q).mean()  # [1]
		self.optimizer_q2.zero_grad(set_to_none=True)
		critic2_loss.backward()
		self.optimizer_q2.step()
		
		# Update the target network, copying all weights and biases in DQN
		if self.perform_polyak_update:
			for target_param, param in zip(self.target_q1.parameters(), self.model_q1.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for target_param, param in zip(self.target_q2.parameters(), self.model_q2.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		else:
			if self.total_steps % self.target_update_freq == 0:
				self.target_q1.load_state_dict(self.model_q1.state_dict())
				self.target_q2.load_state_dict(self.model_q2.state_dict())

		# 3. Calculate Actor Loss
		exp_action = torch.exp(u_diff.detach() * self.iql_temperature)  # [batch_size, 1]
		# take minimum of exp_action and 100.0 to avoid overflow
		exp_action = torch.min(exp_action, torch.tensor(100.0).to(exp_action.device))  # [batch_size, 1]
		# _, action_log_prob = self.get_action(observations, return_log_probs=True)  # [batch_size, 1]
		action_feats = self.model_actor(observations)  # [batch_size, 512]
		action_dist = self.actor_dist(action_feats)
		action_log_prob = action_dist.log_probs(actions)
		actor_loss = -(exp_action * action_log_prob).mean()  # [1]
		self.optimizer_actor.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.optimizer_actor.step()

		self.total_steps += 1

		# create stats dict
		with torch.no_grad():
			loss = value_loss + actor_loss + critic1_loss + critic2_loss
		stats = {
			"loss": loss.item(),
			"value_loss": value_loss.item(),
			"critic1_loss": critic1_loss.item(),
			"critic2_loss": critic2_loss.item(),
			"actor_loss": actor_loss.item(),
			"total_steps": self.total_steps,
		}
		# print(stats["actor_loss"])
		return stats

	@property
	def calculate_eps(self):
		"""
		Calculate epsilon given the current timestep, initial epsilon, end epsilon and decay rate.
		"""
		eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.total_steps / self.eps_decay)
		return eps

	def eval_step(self, observations, eps=0.0, return_log_probs=False):
		"""
		Given an observation, return an action.

		:param observation: the observation for the environment
		:param eps: the epsilon value for epsilon-greedy action selection
		:return: the action for the environment in numpy
		"""
		deterministic = eps == 0.0
		with torch.no_grad():
			action_feats = self.model_actor(observations)
			action_dist = self.actor_dist(action_feats)

			if deterministic:
				action = action_dist.mode()
			else:
				action = action_dist.sample()  # [batch_size, 1]

			action_log_prob = action_dist.log_probs(action)

		if return_log_probs:
			return action.cpu().numpy(), action_log_prob

		return action.cpu().numpy()

	def get_action(self, observations, eps=0.0, return_log_probs=False):
		"""
		Given an observation, return an action.

		:param observation: the observation for the environment
		:param eps: the epsilon value for epsilon-greedy action selection
		:return: the action for the environment in numpy
		"""
		deterministic = eps == 0.0

		action_feats = self.model_actor(observations)  # [batch_size, 512]
		action_dist = self.actor_dist(action_feats)

		if deterministic:
			action = action_dist.mode()
		else:
			action = action_dist.sample()  # [batch_size, 1]

		action_log_prob = action_dist.log_probs(action)
		# st()
		# print(action_log_prob)

		if return_log_probs:
			return action, action_log_prob

		return action

	def save(self, num_epochs, path):
		"""
		Save the model to a given path.

		:param path: the path to save the model
		"""
		save_dict = {
			"actor_state_dict": self.model_actor.state_dict(),
			"actor_dist_state_dict": self.actor_dist.state_dict(),
			"model_v_state_dict": self.model_v.state_dict(),
			"model_q1_state_dict": self.model_q1.state_dict(),
			"model_q2_state_dict": self.model_q2.state_dict(),
			"target_q1_state_dict": self.target_q1.state_dict(),
			"target_q2_state_dict": self.target_q2.state_dict(),
			"optimizer_actor_state_dict": self.optimizer_actor.state_dict(),
			"optimizer_v_state_dict": self.optimizer_v.state_dict(),
			"optimizer_q1_state_dict": self.optimizer_q1.state_dict(),
			"optimizer_q2_state_dict": self.optimizer_q2.state_dict(),
			"total_steps": self.total_steps,
			"curr_epochs": num_epochs,
		}
		torch.save(save_dict, path)
		return

	def load(self, path):
		"""
		Load the model from a given path.

		:param path: the path to load the model
		"""
		checkpoint = torch.load(path)
		self.model_actor.load_state_dict(checkpoint["actor_state_dict"])
		self.actor_dist.load_state_dict(checkpoint["actor_dist_state_dict"])
		self.model_v.load_state_dict(checkpoint["model_v_state_dict"])
		self.model_q1.load_state_dict(checkpoint["model_q1_state_dict"])
		self.model_q2.load_state_dict(checkpoint["model_q2_state_dict"])
		self.target_q1.load_state_dict(checkpoint["target_q1_state_dict"])
		self.target_q2.load_state_dict(checkpoint["target_q2_state_dict"])
		self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
		self.optimizer_v.load_state_dict(checkpoint["optimizer_v_state_dict"])
		self.optimizer_q1.load_state_dict(checkpoint["optimizer_q1_state_dict"])
		self.optimizer_q2.load_state_dict(checkpoint["optimizer_q2_state_dict"])
		self.total_steps = checkpoint["total_steps"]
		self.target_q1.eval()
		self.target_q2.eval()

		return checkpoint["curr_epochs"]
```

procgen/train_scripts/arguments.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

parser = argparse.ArgumentParser(description="Make commands")

parser.add_argument(
    "--action",
    type=str,
    choices=["PRINT", "SAVE", "SLURM"],
    default="PRINT",
    help="PRINT: Print generated python commands out to terminal. "
    + "SAVE: Save generated python commands into a file. "
    + "SLURM: Schedule slurm jobs wtih generated commands.",
)

# Principle arguments
parser.add_argument(
    "--base_config",
    type=str,
    choices=["offline", "online"],
    default="offline",
    help="Base config where parameters are set with default values, and may be replaced by sweeping.",
)
parser.add_argument(
    "--grid_config",
    type=str,
    choices=["bc", "ppo", "bcq", "cql", "iql", "dt", "bct"],
    help="Name of the .json config for hyperparameter search-grid.",
)
parser.add_argument(
    "--num_trials", type=int, default=1, help="Number of seeds to be used for each hyperparameter setting."
)
parser.add_argument(
    "--module_name",
    type=str,
    choices=["offline.train_offline_agent", "online.trainer"],
    default="offline.train_offline_agent",
    help="Name of module to be used in the generated commands. "
    + "The result will be like 'python -m <MODULE_NAME> ...'",
)
parser.add_argument("--start_index", default=0, type=int, help="Starting trial index of xpid runs")
parser.add_argument(
    "--checkpoint",
    action="store_true",
    help="If true, a boolean flag 'resume' will be put in the generated commands, "
    + "which indicates offline training to resume from a given checkpoint.",
)

# Misc
parser.add_argument(
    "--new_line",
    action="store_true",
    help="If true, print generated commands with arguments separated by new line; "
    + "otherwise arguments will be separated by space.",
)

# wandb
parser.add_argument("--wandb_api_key", type=str, default=None, help="wandb api key")
parser.add_argument("--wandb_base_url", type=str, default=None, help="wandb base url")
parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity")
parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")

# Slurm
parser.add_argument("--job_name", default="anyslurm", help="Slurm job name.")
parser.add_argument(
    "--dry_run", default=False, help="If true, a dry run will be performed and NO slurm jobs will be scheduled."
)
```

procgen/train_scripts/cmd_generator.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import itertools
import json
import os
import pathlib
import sys
from typing import Dict, List

import submitit

from train_scripts import arguments
from utils.utils import merge_two_dicts, permutate_params_and_merge


def generate_all_params(grid, defaults, num_trials=1, start_index=0) -> List[Dict[str, any]]:
    grid = merge_two_dicts(
        grid,
        {
            "xpid": [n for n in range(start_index, start_index + num_trials)],
        },
    )
    permutations = permutate_params_and_merge(grid, defaults)
    return [
        merge_two_dicts(
            perm,
            {"seed": perm["seed"] + perm["xpid"]},
        )
        for perm in permutations
    ]


def generate_command(params: Dict[str, any], newlines: bool, xpid_generator, algo: str) -> str:
    if xpid_generator:
        params["xpid"] = xpid_generator(params, algo) + f"_{params['xpid']}"

    separator = " \\\n" if newlines else " "
    header = f"python -m {args.module_name}"
    cmd = [header] + [f"--{k}={vi}" for k, v in params.items() for vi in (v if isinstance(v, list) else [v])]
    return separator.join(cmd)


def generate_slurm_commands(params: Dict[str, any], module_name: str, xpid_generator, algo: str) -> List[str]:
    if xpid_generator:
        params["xpid"] = xpid_generator(params, algo) + f"_{params['xpid']}"

    header = [sys.executable, "-m", module_name]
    args = itertools.chain(
        *[(f"--{k}", str(vi)) for k, v in params.items() for vi in (v if isinstance(v, list) else [v])]
    )
    return header + list(args)


def xpid_from_params(p, algo: str = "") -> str:
    env_prefix = f"{p['env_name']}-{p['distribution_mode']}-{p['num_levels']}"

    """python function which converts long integers into short strings
        Example: 1000000 -> 1M, 1000 -> 1K, etc.
    """

    def short_int(n):
        if n >= 1_000_000:
            return f"{int(n/1_000_000)}M"
        elif n >= 1000:
            return f"{int(n/1000)}K"
        else:
            return f"{n}"

    if "dataset_size" in p:
        env_prefix = f"{env_prefix}-d{short_int(p['dataset_size'])}"

    if algo in ["bc", "cql", "bcq", "iql", "dt", "bct"]:
        algo_prefix = f"{algo}-p{p['percentile']}-lr{p['lr']}-bs{p['batch_size']}-{p['agent_model']}"
        if algo in ["cql", "bcq", "iql"]:
            algo_prefix = f"{algo_prefix}-tau{p['tau']}-tuf{p['target_update_freq']}"
        if algo == "cql":
            algo_prefix = f"{algo_prefix}-a{p['cql_alpha']}"
        elif algo == "bcq":
            assert p["agent_model"] in ["bcq", "bcqresnetbase"]
            algo_prefix = f"{algo_prefix}-t{p['bcq_threshold']}"
            if p["agent_model"] == "bcqresnetbase":
                algo_prefix = f"{algo_prefix}-res"
        elif algo == "iql":
            algo_prefix = f"{algo_prefix}-t{p['iql_temperature']}-e{p['iql_expectile']}"
        elif algo in ["dt", "bct"]:
            algo_prefix = f"{algo_prefix}-cl{p['dt_context_length']}-er{p['dt_eval_ret']}"
    elif algo == "ppo":
        algo_prefix = (
            f"{algo}-lr{p['lr']}-epoch{p['ppo_epoch']}-mb{p['num_mini_batch']}"
            + f"-v{p['value_loss_coef']}-ha{p['entropy_coef']}"
        )
    else:
        algo_prefix = f"{algo}-lr{p['lr']}"
    
    if "early_stop" in p and p['early_stop']:
        algo_prefix = f"{algo_prefix}-es"

    return f"{env_prefix}-{algo_prefix}"


class LaunchExperiments:
    def __init__(self):
        pass

    def launch_experiment_and_remoteenv(self, experiment_args):
        # imports and definition are inside of function because of submitit
        import multiprocessing as mp

        def launch_experiment(experiment_args):
            import subprocess

            subprocess.call(
                generate_slurm_commands(
                    params=experiment_args,
                    module_name=args.module_name,
                    xpid_generator=xpid_from_params,
                    algo=args.grid_config,
                )
            )

        experiment_process = mp.Process(target=launch_experiment, args=[experiment_args])
        self.process = experiment_process
        experiment_process.start()
        experiment_process.join()

    def __call__(self, experiment_args):
        self.launch_experiment_and_remoteenv(experiment_args)

    def checkpoint(self, experiment_args) -> submitit.helpers.DelayedSubmission:
        self.process.terminate()
        return submitit.helpers.DelayedSubmission(LaunchExperiments(), experiment_args)


def schedule_slurm_jobs(all_params: List[Dict[str, any]], job_name: str, dry_run: bool) -> None:
    now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")

    rootdir = os.path.expanduser(f"~/slurm/{job_name}")
    submitit_dir = os.path.expanduser(f"~/slurm/{job_name}/{now}")
    executor = submitit.SlurmExecutor(folder=submitit_dir)
    os.makedirs(submitit_dir, exist_ok=True)

    symlink = os.path.join(rootdir, "latest")
    if os.path.islink(symlink):
        os.remove(symlink)
    if not os.path.exists(symlink):
        os.symlink(submitit_dir, symlink)
        print("Symlinked experiment directory: %s", symlink)

    executor.update_parameters(
        # examples setup
        partition="learnlab",
        # partition="prioritylab",
        comment="Neurips 2023 submission",
        time=1 * 72 * 60,
        nodes=1,
        ntasks_per_node=1,
        # job setup
        job_name=job_name,
        mem="160GB",
        cpus_per_task=10,
        gpus_per_node=1,
        constraint="volta32gb",
        array_parallelism=1024,
    )

    if not dry_run:
        jobs = executor.map_array(LaunchExperiments(), all_params)

        for job in jobs:
            print("Submitted with job id: ", job.job_id)
            print(f"stdout -> {submitit_dir}/{job.job_id}_0_log.out")
            print(f"stderr -> {submitit_dir}/{job.job_id}_0_log.err")

        print(f"Submitted {len(jobs)} jobs! \n")

        print(submitit_dir)


if __name__ == "__main__":
    args = arguments.parser.parse_args()

    # Default Params
    defaults = json.load(
        open(os.path.expandvars(os.path.expanduser(os.path.join("configs", args.base_config, "default.json"))))
    )
    if args.checkpoint:
        defaults["resume"] = True

    if args.wandb_project:
        defaults["wandb_project"] = args.wandb_project

    if args.wandb_base_url:
        defaults["wandb_base_url"] = args.wandb_base_url
    if args.wandb_api_key:
        defaults["wandb_api_key"] = args.wandb_api_key
    if args.wandb_entity:
        defaults["wandb_entity"] = args.wandb_entity

    # Generate all parameter combinations within grid, using defaults for fixed params
    config = json.load(
        open(
            os.path.expandvars(
                os.path.expanduser(os.path.join("configs", args.base_config, "grids", args.grid_config + ".json"))
            )
        )
    )
    all_params = generate_all_params(config["grid"], defaults, args.num_trials, args.start_index)
    print(f"About to generate {len(all_params)} commands! \n")

    # Action
    if args.action == "PRINT":
        cmds = [generate_command(params, args.new_line, xpid_from_params, args.grid_config) for params in all_params]

        print("Generated Commands: \n")
        [print(f"{cmd} \n") for cmd in cmds]

    elif args.action == "SAVE":
        cmds = [generate_command(params, args.new_line, xpid_from_params, args.grid_config) for params in all_params]

        filename = f"{args.grid_config}_commands.txt"
        pathlib.Path(filename).touch()
        with open(filename, "w") as f:
            [f.write(f"{cmd} \n\n") for cmd in cmds]

        print(f"Generated commands are stored in {filename}.")

    elif args.action == "SLURM":
        schedule_slurm_jobs(all_params, args.job_name, args.dry_run)
```

procgen/train_scripts/make_cmd.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from typing import List

from utils.utils import permutate_params_and_merge


def generate_train_cmds(
    params,
    num_trials=1,
    start_index=0,
    newlines=False,
    xpid_generator=None,
    algo="",
    xpid_prefix="",
    is_single=False,
) -> List[str]:
    separator = " \\\n" if newlines else " "

    cmds = []

    if xpid_generator:
        params["xpid"] = xpid_generator(params, xpid_prefix, algo, is_single)

    for t in range(num_trials):
        trial_idx = t + start_index
        params["seed"] += trial_idx

        cmd = [f"python -m {args.module_name}"] + [
            f"--{k}={vi}_{trial_idx}" if k == "xpid" else f"--{k}={vi}"
            for k, v in params.items()
            for vi in (v if isinstance(v, list) else [v])
        ]

        cmds.append(separator.join(cmd))

    return cmds


def parse_args():
    parser = argparse.ArgumentParser(description="Make commands")

    # Principle arguments
    parser.add_argument(
        "--base_config",
        type=str,
        choices=["offline", "online"],
        default="offline",
        help="Base config where parameters are set with default values, and may be replaced by sweeping.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="grid",
        help="Directory where the configs are present",
    )
    parser.add_argument(
        "--grid_config",
        type=str,
        help="Name of the .json config for hyperparameter search-grid.",
    )
    parser.add_argument(
        "--num_trials", type=int, default=1, help="Number of seeds to be used for each hyperparameter setting."
    )
    parser.add_argument(
        "--module_name",
        type=str,
        choices=["offline.train_offline_agent", "online.trainer", "offline.single_level_train_offline_agent"],
        default="offline.train_offline_agent",
        help="Name of module to be used in the generated commands. "
        + "The result will be like 'python -m <MODULE_NAME> ...'",
    )
    parser.add_argument("--start_index", default=0, type=int, help="Starting trial index of xpid runs")
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="If true, a boolean flag 'resume' will be put in the generated commands, "
        + "which indicates offline training to resume from a given checkpoint.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="single level training"
    )

    # Misc
    parser.add_argument(
        "--new_line",
        action="store_true",
        help="If true, print generated commands with arguments separated by new line; "
        + "otherwise arguments will be separated by space.",
    )
    parser.add_argument("--count", action="store_true", help="If true, print number of generated commands at the end.")

    # wandb
    parser.add_argument("--wandb_base_url", type=str, default=None, help="wandb base url")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="wandb api key")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity")
    parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")

    return parser.parse_args()


def xpid_from_params(p, prefix="", algo="", is_single=False):
    env_prefix = f"{p['env_name']}-{p['distribution_mode']}-{p['num_levels']}"
    
    """python function which converts long integers into short strings
        Example: 1000000 -> 1M, 1000 -> 1K, etc.
    """
    def short_int(n):
        if n >= 1000000:
            return f"{int(n/1000000)}M"
        elif n >= 1000:
            return f"{int(n/1000)}K"
        else:
            return f"{n}"
        
    if "dataset_size" in p:
        env_prefix = f"{env_prefix}-d{short_int(p['dataset_size'])}"

    if algo in ["bc", "cql", "bcq", "iql", "dt", "bct"]:
        algo_prefix = f"{algo}-p{p['percentile']}-lr{p['lr']}-bs{p['batch_size']}-{p['agent_model']}"
        if algo in ["cql", "bcq", "iql"]:
            algo_prefix = f"{algo_prefix}-tuf{p['target_update_freq']}"
            if p['perform_polyak_update']:
                algo_prefix = f"{algo_prefix}-polyak-tau{p['tau']}"
        if algo == "cql":
            algo_prefix = f"{algo_prefix}-a{p['cql_alpha']}"
        elif algo == "bcq":
            assert p["agent_model"] in ["bcq", "bcqresnetbase"]
            algo_prefix = f"{algo_prefix}-t{p['bcq_threshold']}"
            if p["agent_model"] == "bcqresnetbase":
                algo_prefix = f"{algo_prefix}-res"
        elif algo == "iql":
            algo_prefix = f"{algo_prefix}-t{p['iql_temperature']}-e{p['iql_expectile']}"
        elif algo in ["dt", "bct"]:
            algo_prefix = f"{algo_prefix}-cl{p['dt_context_length']}-er{p['dt_eval_ret']}"
    elif algo == "ppo":
        algo_prefix = (
            f"{algo}-lr{p['lr']}-epoch{p['ppo_epoch']}-mb{p['num_mini_batch']}"
            + f"-v{p['value_loss_coef']}-ha{p['entropy_coef']}"
        )
    else:
        algo_prefix = f"{algo}-lr{p['lr']}"
        
    if "early_stop" in p and p['early_stop']:
        algo_prefix = f"{algo_prefix}-es"
        
    if is_single:
        algo_prefix = f"{algo_prefix}-single"
        if p['capacity_type']=="transitions":
            algo_prefix = f"{algo_prefix}-t"
        elif p['capacity_type']=="episodes":
            algo_prefix = f"{algo_prefix}-e"
        algo_prefix = f"{algo_prefix}-{p['threshold_metric']}"

    return f"{env_prefix}-{algo_prefix}"


if __name__ == "__main__":
    args = parse_args()

    # Default Params
    defaults = json.load(
        open(os.path.expandvars(os.path.expanduser(os.path.join("configs", args.base_config, "default.json"))))
    )
    if args.checkpoint:
        defaults["resume"] = True

    if args.wandb_project:
        defaults["wandb_project"] = args.wandb_project

    if args.wandb_base_url:
        defaults["wandb_base_url"] = args.wandb_base_url
    if args.wandb_api_key:
        defaults["wandb_api_key"] = args.wandb_api_key
    if args.wandb_entity:
        defaults["wandb_entity"] = args.wandb_entity

    # Generate all parameter combinations within grid, using defaults for fixed params
    config = json.load(
        open(
            os.path.expandvars(
                os.path.expanduser(os.path.join("configs", args.base_config, args.dir, args.grid_config + ".json"))
            )
        )
    )
    all_params = permutate_params_and_merge(config["grid"], defaults=defaults)

    # Print all commands
    xpid_prefix = "" if "xpid_prefix" not in config else config["xpid_prefix"]
    for p in all_params:
        cmds = generate_train_cmds(
            p,
            num_trials=args.num_trials,
            start_index=args.start_index,
            newlines=args.new_line,
            xpid_generator=xpid_from_params,
            algo=args.grid_config,
            xpid_prefix=xpid_prefix,
            is_single=args.single,
        )

        for c in cmds:
            print(c + "\n")

    if args.count:
        print(f"Generated {len(all_params) * args.num_trials} commands.")
```

procgen/train_scripts/slurm.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import sys

import submitit
from absl import app, flags
from coolname import generate_slug

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "~/cmds.txt", "Path to list of commands to run.")
flags.DEFINE_string("name", "anyslurm", "Experiment name.")
flags.DEFINE_boolean("debug", False, "Only debugging output.")
flags.DEFINE_string("module_name", "offline", "module name tyoe")


def arg2str(k, v):
    if isinstance(v, bool):
        if v:
            return ("--%s" % k,)
        else:
            return ""
    else:
        return ("--%s" % k, str(v))


class LaunchExperiments:
    def __init__(self, module_name: str="offline"):
        self.module_name = module_name

    def launch_experiment_and_remotenv(self, experiment_args):
        # imports and definition are inside of function because of submitit
        import multiprocessing as mp

        def launch_experiment(experiment_args):
            import itertools
            import subprocess

            python_exec = sys.executable
            args = itertools.chain(*[arg2str(k, v) for k, v in experiment_args.items()])

            if self.module_name == "offline":
                subprocess.call([python_exec, "-m", "offline.train_offline_agent"] + list(args))
            elif self.module_name == "eval":
                subprocess.call([python_exec, "-m", "offline.evaluate_offline_agent"] + list(args))
            elif self.module_name == "single":
                subprocess.call([python_exec, "-m", "offline.single_level_train_offline_agent"] + list(args))
            elif self.module_name == "data_collection":
                subprocess.call([python_exec, "-m", "online.data_collector"] + list(args))
            else:
                subprocess.call([python_exec, "-m", "online.trainer"] + list(args))

        experiment_process = mp.Process(target=launch_experiment, args=[experiment_args])
        self.process = experiment_process
        experiment_process.start()
        experiment_process.join()

    def __call__(self, experiment_args):
        self.launch_experiment_and_remotenv(experiment_args)

    def checkpoint(self, experiment_args) -> submitit.helpers.DelayedSubmission:
        self.process.terminate()
        experiment_args["checkpoint"] = True
        return submitit.helpers.DelayedSubmission(LaunchExperiments(self.module_name), experiment_args)


def main(argv):
    now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")

    rootdir = os.path.expanduser(f"~/slurm/{FLAGS.name}")
    submitit_dir = os.path.expanduser(f"~/slurm/{FLAGS.name}/{now}")
    executor = submitit.SlurmExecutor(folder=submitit_dir)
    os.makedirs(submitit_dir, exist_ok=True)

    symlink = os.path.join(rootdir, "latest")
    if os.path.islink(symlink):
        os.remove(symlink)
    if not os.path.exists(symlink):
        os.symlink(submitit_dir, symlink)
        print("Symlinked experiment directory: %s", symlink)

    all_args = list()

    with open(os.path.expanduser(FLAGS.path), "r") as f:
        cmds = "".join(f.readlines()).split("\n\n")
        cmds = [cmd.split("\\\n")[1:] for cmd in cmds]
        cmds = [cmd for cmd in cmds if len(cmd) > 0]
        for line in cmds:
            le_args = dict()
            for pair in line:
                key, val = pair.strip()[2:].split("=")
                le_args[key] = val
            if "xpid" not in le_args and FLAGS.module_name != "data_collection":
                le_args["xpid"] = generate_slug()

            all_args.append(le_args)

    executor.update_parameters(
        # examples setup
        time=1 * 72 * 60,
        nodes=1,
        ntasks_per_node=1,
        # job setup
        job_name=FLAGS.name,
        mem="512GB",
        cpus_per_task=10,
        gpus_per_node=1,
        constraint="volta32gb",
        array_parallelism=480,
    )

    print("\nAbout to submit", len(all_args), "jobs")

    if not FLAGS.debug:
        job = executor.map_array(LaunchExperiments(module_name=FLAGS.module_name), all_args)

        for j in job:
            print("Submitted with job id: ", j.job_id)
            print(f"stdout -> {submitit_dir}/{j.job_id}_0_log.out")
            print(f"stderr -> {submitit_dir}/{j.job_id}_0_log.err")

        print(f"Submitted {len(job)} jobs!")

        print()
        print(submitit_dir)


if __name__ == "__main__":
    app.run(main)
```

