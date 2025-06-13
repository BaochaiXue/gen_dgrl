./procgen/download.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import functools
import multiprocessing as mp
import os
import shutil
import typing as tp
from argparse import ArgumentParser

import requests
from tqdm import tqdm

BASE_URL = "https://dl.fbaipublicfiles.com/DGRL/"

ENV_NAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]


def download_dataset(
    category_name: str,
    download_folder: str,
    n_download_workers: int = 4,
    n_extract_workers: int = 4,
    clear_archives_after_unpacking: bool = False,
    skip_downloaded_archives: bool = True,
):
    """
    Downloads and unpacks the dataset.

    Note: The script will make a folder `<download_folder>/_in_progress`, which
        stores files whose download is in progress. The folder can be safely deleted
        the download is finished.

    Args:
        category_name: A category in the given dataset.
        download_folder: A local target folder for downloading the
            the dataset files.
        n_download_workers: The number of parallel workers
            for downloading the dataset files.
        n_extract_workers: The number of parallel workers
            for extracting the dataset files.
        clear_archives_after_unpacking: Delete the unnecessary downloaded archive files
            after unpacking.
        skip_downloaded_archives: Skip re-downloading already downloaded archives.
    """

    if not os.path.isdir(download_folder):
        raise ValueError(
            "Please specify `download_folder` with a valid path to a target folder"
            + " for downloading the dataset."
            + f" {download_folder} does not exist."
        )

    links = _build_urls_with_category_name(category_name)
    data_links = [(category_name, _fetch_file_name_from_link(link), link) for link in links]
    print(f"Will download {len(data_links)} files from the following links for {category_name}: {links}")

    print("Downloading ...")
    with mp.Pool(processes=n_download_workers) as download_pool:
        download_ok = {}
        for link_name, ok in tqdm(
            download_pool.imap(
                functools.partial(
                    _download_category_file,
                    download_folder,
                    skip_downloaded_archives,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            download_ok[link_name] = ok

        if not all(download_ok.values()):
            not_ok_links = [n for n, ok in download_ok.items() if not ok]
            not_ok_links_str = "\n".join(not_ok_links)
            raise AssertionError(
                "There are errors when downloading the following files:\n"
                + not_ok_links_str
                + "\n"
                + "This is most likely due to a network failure."
                + " Please restart the download script."
            )

    print(f"Extracting {len(data_links)} dataset files ...")
    with mp.Pool(processes=n_extract_workers) as extract_pool:
        for _ in tqdm(
            extract_pool.imap(
                functools.partial(
                    _unpack_category_file,
                    download_folder,
                    clear_archives_after_unpacking,
                ),
                data_links,
            ),
            total=len(data_links),
        ):
            pass

    print("Done")


def build_arg_parser(
    dataset_name: str = "Procgen",
) -> ArgumentParser:
    parser = ArgumentParser(description=f"Download the {dataset_name} dataset.")
    parser.add_argument(
        "--download_folder",
        type=str,
        required=True,
        help="A local target folder for downloading the the dataset files.",
    )
    parser.add_argument(
        "--category_name",
        type=str,
        required=True,
        choices=["1M_E", "1M_S", "10M", "25M","level_1_S","level_1_E","level_40_S","level_40_E"],
        help="Category name for Procgen environment, based on number of transitions: "
        + "1M, 10M, and 25M. Only data of 1M transitions has expert (E) and suboptimal (S) option.",
    )
    parser.add_argument(
        "--n_download_workers",
        type=int,
        default=4,
        help="The number of parallel workers for downloading the dataset files.",
    )
    parser.add_argument(
        "--n_extract_workers",
        type=int,
        default=4,
        help="The number of parallel workers for extracting the dataset files.",
    )
    parser.add_argument(
        "--clear_archives_after_unpacking",
        action="store_true",
        default=False,
        help="Delete the unnecessary downloaded archive files after unpacking.",
    )
    parser.add_argument(
        "--redownload_existing_archives",
        action="store_true",
        default=False,
        help="Redownload the already-downloaded archives.",
    )

    return parser


def _build_urls_with_category_name(category_name: str) -> tp.List[str]:
    if category_name in ["level_1_E", "level_1_S", "level_40_E", "level_40_S"]:
        return [
            os.path.join(BASE_URL, _convert_category_name(category_name))
        ]
    else:
        return [
            os.path.join(BASE_URL, _convert_category_name(category_name), f"{env_name}.tar.xz") for env_name in ENV_NAMES
        ]


def _convert_category_name(category_name: str) -> str:
    if category_name == "1M_E":
        return "1M/expert"
    elif category_name == "1M_S":
        return "1M/suboptimal"
    elif category_name == "10M":
        return "10M"
    elif category_name == "25M":
        return "25M"
    elif category_name == "level_1_S":
        return "100k_procgen_dataset_1_suboptimal.tar"
    elif category_name == "level_40_S":
        return "100k_procgen_dataset_40_suboptimal.tar"
    elif category_name == "level_1_E":
        return "100k_procgen_dataset_1.tar"
    elif category_name == "level_40_E":
        return "100k_procgen_dataset_40.tar"
    else:
        raise ValueError(f"Unrecognized category name {category_name}!")


def _fetch_file_name_from_link(url: str) -> str:
    return os.path.split(url)[-1]


def _unpack_category_file(
    download_folder: str,
    clear_archive: bool,
    link: str,
):
    _, file_name, _ = link
    file_path = os.path.join(download_folder, file_name)
    print(f"Unpacking dataset file {file_path} ({file_name}) to {download_folder}.")
    shutil.unpack_archive(file_path, download_folder)
    if clear_archive:
        os.remove(file_path)


def _download_category_file(
    download_folder: str,
    skip_downloaded_files: bool,
    link: str,
):
    _, file_name, url = link
    file_path = os.path.join(download_folder, file_name)

    if skip_downloaded_files and os.path.isfile(file_path):
        print(f"Skipping {file_path}, already downloaded!")
        return file_name, True

    in_progress_folder = os.path.join(download_folder, "_in_progress")
    os.makedirs(in_progress_folder, exist_ok=True)
    in_progress_file_path = os.path.join(in_progress_folder, file_name)

    print(f"Downloading dataset file {file_name} ({url}) to {in_progress_file_path}.")
    _download_with_progress_bar(url, in_progress_file_path)

    os.rename(in_progress_file_path, file_path)
    return file_name, True


def _download_with_progress_bar(url: str, file_path: str):
    # taken from https://stackoverflow.com/a/62113293/986477
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(file_path, "wb") as file, tqdm(
        desc=file_path,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


if __name__ == "__main__":
    parser = build_arg_parser("Procgen")
    args = parser.parse_args()
    download_dataset(
        args.category_name,
        str(args.download_folder),
        n_download_workers=int(args.n_download_workers),
        n_extract_workers=int(args.n_extract_workers),
        clear_archives_after_unpacking=bool(args.clear_archives_after_unpacking),
        skip_downloaded_archives=not bool(args.redownload_existing_archives),
    )
```

./procgen/offline/agents/__init__.py
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

./procgen/offline/agents/bc.py
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

./procgen/offline/agents/bcq.py
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

./procgen/offline/agents/ddqn_cql.py
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

./procgen/offline/agents/dt.py
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

./procgen/offline/agents/iql.py
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

./procgen/offline/arguments.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from utils.utils import str2bool

parser = argparse.ArgumentParser(description="Train offline agents")
parser.add_argument("--algo", type=str, default="bc", choices=["bc", "cql", "dt", "bct", "bcq", "offlinedqn", "iql", "xql"], help="Algorithm to train")
parser.add_argument("--dataset", type=str, default="data/dataset.hdf5", help="Path to dataset")
parser.add_argument("--percentile", type=float, default=1.0, help="percentile for top% training")
parser.add_argument("--dataset_size", type=int, default=1000000, help="Size of dataset")
parser.add_argument("--early_stop", type=str2bool, default=False, help="Use early stopping")

# Model
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--hidden_size", type=int, default=64, help="Size of hidden layers")
parser.add_argument("--agent_model", type=str, default="dqn", choices=["dqn", "bcq", "pporesnetbase", "pporesnet20", "bcqresnetbase"], help="Agent model")
parser.add_argument("--save_path", type=str, default="data/bc.pt", help="Path to save model")
parser.add_argument("--resume", type=str2bool, default=False, help="Resume training")
parser.add_argument("--deterministic", type=str2bool, default=False, help="Sample actions deterministically")
parser.add_argument("--xpid", type=str, default=None, help="experiment name")
parser.add_argument("--eval_eps", type=float, default=0.001, help="epsilon for evaluation")

# Environment
parser.add_argument("--env_name", type=str, default="bigfish", help="Name of environment")
parser.add_argument("--seed", type=int, default=88, help="experiment seed")
parser.add_argument("--num_levels", type=int, default=200, help="number of training levels used in procgen")
parser.add_argument("--distribution_mode", type=str, default="easy", help="Distribution mode of procgen levels")
parser.add_argument("--eval_freq", type=int, default=10, help="frequency for eval")
parser.add_argument("--ckpt_freq", type=int, default=10, help="frequency for checkpointing")

# DDQN
parser.add_argument("--target_update_freq", type=int, default=1000, help="frequency for target network update")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--tau", type=float, default=0.005, help="soft update factor")
parser.add_argument("--buffer_size", type=int, default=1000000, help="size of replay buffer")
parser.add_argument("--eps_start", type=float, default=1.0, help="epsilon start value")
parser.add_argument("--eps_end", type=float, default=0.01, help="epsilon end value")
parser.add_argument("--eps_decay", type=int, default=1000000, help="epsilon decay rate")
parser.add_argument("--perform_polyak_update", type=str2bool, default=False, help="whether to use polyak average or directly copy model weights")

# CQL
parser.add_argument("--cql_alpha", type=float, default=1.0, help="CQL Loss alpha")

# BCQ
parser.add_argument("--bcq_threshold", type=float, default=0.3, help="BCQ threshold for action selection")

# IQL
parser.add_argument("--iql_temperature", type=float, default=0.1, help="IQL temperature for action selection")
parser.add_argument("--iql_expectile", type=float, default=0.8, help="IQL Expectile Loss")

# DT
parser.add_argument("--dt_context_length", type=int, default=128, help="context length for the agent")
parser.add_argument("--grad_norm_clip", type=float, default=0.1, help="gradient norm clip for the agent")
parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay only applied on matmul weights")
parser.add_argument("--lr_decay", type=str2bool, default=True, help="learning rate decay with linear warmup followed by cosine decay to 10% of original")
parser.add_argument("--warmup_tokens", type=int, default=10000, help="warmup tokens")
parser.add_argument("--dt_rtg_noise_prob", type=float, default=0.0, help="noise probability for RTGs")
parser.add_argument("--dt_eval_ret", type=int, default=0, help="evaluation return to go. if > 0, then eval rtg = args.dt_eval_ret * max_return in the dataset")

# Single Level Training
parser.add_argument("--capacity_type", type=str, default="transitions", choices=["transitions", "episodes"], help="capacity type")
parser.add_argument("--threshold_metric", type=str, default="median", choices=["percentile", "median"], help="threshold metric")
```

./procgen/offline/dataloader.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import os
import gc
from itertools import accumulate
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Any, Callable, Iterable, Tuple, Union
import numpy as np
import torch
import random

from utils.utils import DatasetItemType


def load_episode(path) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        episode['observations'] = episode['observations'].astype(np.uint8)
        episode['rewards'] = episode['rewards'].astype(np.float)
        return episode


def compute_episode_length(episode: Dict[str, np.ndarray]) -> int:
    return len(episode[DatasetItemType.ACTIONS.value])


def fetch_return_from_path(name: Path) -> float:
    return float(name.stem.split("_")[-1])


class OfflineDataset(torch.utils.data.Dataset):
    """
    Load episodes from files and sample a batch (obs, next_obs, action, reward, done)
    from one of the loaded episodes.
    """

    _capacity: int
    _episodes_dir_path: List[str]
    _episodes: List[Dict[str, np.ndarray]]
    _loaded: bool
    _size: int
    _num_transitions: int
    _percentile: float
    _zero_out_last_obs: bool
    _episode_lengths: List[int]
    _capacity_type: str
    _specific_level_seed: Optional[int]
    _max_episodes: Optional[int]
    

    def __init__(
        self, capacity: int, episodes_dir_path: List[str], percentile: float = 1.0, zero_out_last_obs: bool = True,
        capacity_type: str = 'transitions', specific_level_seed: Optional[int] = None, max_episodes: Optional[int] = None
    ) -> None:
        self._capacity = capacity
        self._episodes_dir_path = (
            [directory for directory in episodes_dir_path]
            if isinstance(episodes_dir_path, list)
            else [episodes_dir_path]
        )
        self._episodes = []
        self._loaded = False
        self._size = 0
        self._percentile = percentile
        self._zero_out_last_obs = zero_out_last_obs
        self._specific_level_seed = specific_level_seed
        self._capacity_type = capacity_type
        if self._capacity_type == 'episodes':
            assert max_episodes is not None, "max_episodes must be specified when capacity_type is 'episodes'"
        elif self._capacity_type == 'transitions':
            assert max_episodes is None, "max_episodes must be None when capacity_type is 'transitions'"
        self._max_episodes = max_episodes
        self._sort_by_return_and_load_by_percentile()
        
    def _calc_average_return(self) -> float:
        # calculate average return across all episodes
        rewards = []
        for episode in self._episodes:
            rewards.append(np.sum(episode[DatasetItemType.REWARDS.value]))
        return np.mean(rewards)
            

    def _sort_by_return_and_load_by_percentile(self) -> None:
        if self._loaded is True:
            return

        episode_filenames = sorted(
            [f.path for directory in self._episodes_dir_path for f in os.scandir(directory)],
            key=lambda path: OfflineDataset._fetch_return_from_path(path),
            reverse=True,
        )
        print(f"[DEBUG] Total number of episodes: {len(episode_filenames)}.")

        num_transitions = int(self._capacity * self._percentile)
        print(
            f"[DEBUG] Capacity: {self._capacity}. "
            + f"Loading {num_transitions} ({100 * self._percentile}% of {self._capacity}) transitions ..."
        )

        # Store all episodes (capped by _capacity and _percentile) into _episodes
        for name in episode_filenames:
            if self._specific_level_seed is not None:
                level_seed = int(name.split('/')[-1].split('_')[-2])
                if level_seed != self._specific_level_seed:
                    continue
            episode = load_episode(name)
            curr_episode_len = compute_episode_length(episode)
            if curr_episode_len <= 0:
                # Filter out invalid episodes
                continue

            self._episodes.append(episode)
            self._size += curr_episode_len
            if self._capacity_type  == 'transitions' and self._size > num_transitions:
                break
            elif self._capacity_type == 'episodes' and len(self._episodes) >= self._max_episodes:
                break

        print(f"[DEBUG] Loaded {len(self._episodes)} episodes with {self._size} transitions in total!")
        self._loaded = True
        self._num_transitions = min(num_transitions, self._size)

        # _episode_lengths store the cumulated sum of lengths of episodes that are before the current one (included).
        # As if all stored episodes are concatenated.
        self._episode_lengths = list(accumulate(compute_episode_length(episode) for episode in self._episodes))

    @staticmethod
    def _fetch_return_from_path(path: str) -> float:
        """
        Example:
        >>> path = '/my_documents/model_1_3.5.pt'
        >>> score = fetch_return_from_path(path)
        >>> score
        3.5
        """
        return float(Path(path).stem.split("_")[-1])

    def __len__(self) -> int:
        return self._num_transitions

    def __getitem__(self, index):
        """
        Example:
            Suppose we have 3 episodes with length: 5, 3, 7. The capacity of the dataset is 10, percentile 100%.

            Then we will store those 3 episodes, and _episode_lengths == [5, 5+3, 5+3+7].

            The sample index will be 0 <= index <= 9, given the capacity as 10.

            If index == 6, we know index >= 5 and index < 8, thus it should be in the middle episode, and its offset
            to the beginning of that episode is (6-5) == 1.
        """
        # Use binary search to find out which episode whose transitions match the index.
        episode_idx = bisect.bisect_right(self._episode_lengths, index)
        curr = self._episodes[episode_idx]
        # Compute the offset within the episode
        offset_in_episode = index - (0 if episode_idx == 0 else self._episode_lengths[episode_idx - 1])

        # Fetch corresponding state-action tuple.
        obs = curr[DatasetItemType.OBSERVATIONS.value][offset_in_episode]
        next_obs = (
            # If the episode is completed, the next_obs is a zero vector
            np.zeros_like(obs)
            if self._zero_out_last_obs and curr[DatasetItemType.DONES.value][offset_in_episode]
            else curr[DatasetItemType.OBSERVATIONS.value][offset_in_episode + 1]
        )
        actions = curr[DatasetItemType.ACTIONS.value][offset_in_episode]
        rewards = curr[DatasetItemType.REWARDS.value][offset_in_episode]
        dones = curr[DatasetItemType.DONES.value][offset_in_episode] != 0

        return (obs, actions, rewards, next_obs, dones)


class OfflineDTDataset:
    """
    A dataset designed for Decision Transformer.
    """

    _capacity: int
    _episodes_dir_path: List[Path]
    _episodes: List[Dict[str, np.ndarray]]
    _loaded: bool
    _size: int
    _rtg_noise_prob: float
    _short_traj_count: int
    _context_len: int
    _percentile: float
    _specific_level_seed: Optional[int]
    _capacity_type: str

    def __init__(
        self,
        capacity: int,
        episodes_dir_path: List[str],
        context_len: int,
        rtg_noise_prob: float,
        percentile: float = 1.0,
        specific_level_seed: Optional[int] = None, 
        capacity_type: str = 'transitions'
    ) -> None:
        self._capacity = capacity
        self._episodes_dir_path = (
            [Path(directory) for directory in episodes_dir_path]
            if isinstance(episodes_dir_path, list)
            else [Path(episodes_dir_path)]
        )
        self.context_len = context_len
        self._block_size = context_len * 3
        self._rtg_noise_prob = rtg_noise_prob
        if self._rtg_noise_prob > 0:
            print("[DEBUG] Using RTG noise with probability", self._rtg_noise_prob)
        self._short_traj_count = 0

        self._episodes = []
        self._loaded = False
        self._size = 0
        self._percentile = percentile
        self._max_return = -float("inf")
        
        self._specific_level_seed = specific_level_seed
        self._capacity_type = capacity_type

        self._load()
        self._reassemble()

    def _load(self) -> None:
        if self._loaded is True:
            return

        episode_filenames = [
            name for gens in (directory.rglob("*.npz") for directory in self._episodes_dir_path) for name in gens
        ]
        
        
        # Randomly shuffle episode files
        random.shuffle(episode_filenames)
        
        print(f"[DEBUG] Total number of episodes: {len(episode_filenames)}.")
        print(f"[DEBUG] Capacity: {self._capacity}. Loading {self._capacity} {self._capacity_type} ...")

        # Initialize count of tuples
        tuples_count = 0

        # Target number of tuples
        target_tuples = self._capacity
        
        for name in tqdm(episode_filenames):
            if self._specific_level_seed is not None:
                level_seed = int(str(name).split('/')[-1].split('_')[-2])
                if level_seed != self._specific_level_seed:
                    continue
            # episode = load_episode(name)
            # curr_episode_len = compute_episode_length(episode)
            curr_episode_len = int(name.stem.split('_')[-3])
            if curr_episode_len <= 3:
                # Filter out invalid episodes
                continue

            self._episodes.append(name)
            tuples_count += curr_episode_len
            current_return = fetch_return_from_path(name)
            if tuples_count >= target_tuples:
                break
            self._max_return = max(self._max_return, current_return)

            # if self._size > self._capacity:
            #     break

        self._size = tuples_count
        print(
            f"[DEBUG] Loaded {len(self._episodes)} episodes with {self._size} transitions in total!"
            + f"Maximum return: {self._max_return}"
        )
        self._loaded = True

    def _reassemble(self) -> None:
        # Get shapes
        first = load_episode(self._episodes[0])
        first_obs = first[DatasetItemType.OBSERVATIONS.value][0]
        first_action = first[DatasetItemType.ACTIONS.value][0]
        first_reward = first[DatasetItemType.REWARDS.value][0]
        first_done = first[DatasetItemType.DONES.value][0]

        # Pick top percentile and reassemble
        self._obs = np.empty(shape=(self._size, *first_obs.shape), dtype=first_obs.dtype)
        self._actions = np.empty(shape=(self._size,), dtype=first_action.dtype)
        self._rewards = np.empty(shape=(self._size,), dtype=first_reward.dtype)
        self._return_to_gos = np.empty(shape=(self._size,), dtype=first_reward.dtype)
        self._dones = np.empty(shape=(self._size, *first_done.shape), dtype=np.bool_)
        self._timesteps = np.empty(shape=(self._size,), dtype=np.int)

        print(f"[DEBUG] Assembling {self._size} transitions ...")

        idx = 0
        for curr_file in self._episodes:
            curr = load_episode(curr_file)
            length = int(curr_file.stem.split('_')[-3]) #compute_episode_length(curr)

            end = min(idx + length, self._size)

            self._obs[idx:end, :] = curr[DatasetItemType.OBSERVATIONS.value][: end - idx]
            self._actions[idx:end] = curr[DatasetItemType.ACTIONS.value][: end - idx].squeeze(-1)
            self._rewards[idx:end] = curr[DatasetItemType.REWARDS.value][: end - idx].squeeze(-1)
            self._return_to_gos[idx:end] = OfflineDTDataset.compute_return_to_go(curr)[: end - idx].squeeze(-1)
            self._dones[idx:end] = curr[DatasetItemType.DONES.value][: end - idx]
            assert length == (end - idx)
            self._timesteps[idx:end] = np.arange(length)
            idx += length
            if idx >= self._size:
                print(f"[INFO] Finished assembling {self._size} transitions!")
                break

        self._sanity_check()

        self._min_rtgs = self._return_to_gos.min()
        self._max_rtgs = self._return_to_gos.max()
        self.vocab_size = max(self._actions) + 2
        self._padding_action = self.vocab_size - 1
        self._done_idxs = np.argwhere(self._dones).squeeze(-1) + 1

    def _sanity_check(self) -> None:
        """
        Verify the actions in the last episode correspond to the assembled data.
        """
        idx = 0
        for curr in self._episodes:
            length = int(curr.stem.split('_')[-3]) #compute_episode_length(curr)

            if idx + length >= self._size:
                break

            idx += length

        last_episode = load_episode(self._episodes[-1])
        assert np.allclose(
            self._actions[idx:], last_episode[DatasetItemType.ACTIONS.value][: self._size - idx].squeeze(-1)
        ), "[ERROR] Sanity Check fails. Check the assembling logic of transitions."

        print("[DEBUG] Sanity Check succeeded.")

    def get_max_return(self, multiplier: int) -> float:
        return self._max_return * multiplier

    @staticmethod
    def compute_return_to_go(episode: Dict[str, np.ndarray]) -> np.ndarray:
        return np.cumsum(episode[DatasetItemType.REWARDS.value][::-1], axis=0)[::-1]
    
    def _calc_average_return(self) -> float:
        # calculate average return across all episodes
        rewards = []
        for episode in self._episodes:
            # rewards.append(np.sum(episode[DatasetItemType.REWARDS.value]))
            rewards.append(fetch_return_from_path(episode))
        return np.mean(rewards)

    def __len__(self):
        return len(self._obs) - self._block_size

    def __getitem__(self, idx):
        block_size = self._block_size // 3
        done_idx = idx + block_size
        for i, j in enumerate(self._done_idxs):
            if j > idx:  # first done_idx greater than idx
                done_idx = min(int(j), done_idx)
                break

        if i > 0:
            curr_ep_len = self._done_idxs[i] - self._done_idxs[i - 1]
        else:
            curr_ep_len = self._done_idxs[0]

        if curr_ep_len < block_size:
            print("[DEBUG] The trajectory is shorter than the context size. Hence padding this trajectory.....")
            self._short_traj_count += 1
            if i > 0:
                # find episode start index
                idx = self._done_idxs[i - 1]
            else:
                idx = 0
        else:
            idx = done_idx - block_size
        states = torch.tensor(np.array(self._obs[idx:done_idx]), dtype=torch.float32)  # (size, 3, 64, 64)
        # pad states if episode_len  = (done_idx - idx) < block_size
        states = torch.cat(
            [states, torch.zeros(block_size - states.shape[0], *states.shape[1:])], dim=0
        )  # (block_size, 3, 64, 64)
        states = states.reshape(block_size, -1)  # (block_size, 3*64*64)
        states = states / 255.0
        actions = torch.tensor(self._actions[idx:done_idx], dtype=torch.long).unsqueeze(1)  # (block_size, 1)
        actions = torch.cat(
            [actions, torch.zeros(block_size - actions.shape[0], actions.shape[1]) + self._padding_action], dim=0
        )
        rtgs = self._return_to_gos[idx:done_idx]  # (context size,)
        rtgs = np.pad(rtgs, (0, block_size - rtgs.shape[0]), mode="constant", constant_values=0)
        if self._rtg_noise_prob > 0:
            binary_mask = np.where(np.random.rand(done_idx - idx) < self._rtg_noise_prob)
            # binary_mask[rtgs < 0] = 1
            # rtgs = np.multiply(rtgs, binary_mask)
            random_rtgs = np.random.randint(self._min_rtgs, self._max_rtgs, size=rtgs.shape)
            rtgs[binary_mask] = random_rtgs[binary_mask]

        rtgs = torch.tensor(rtgs, dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self._timesteps[idx : idx + 1], dtype=torch.int64).unsqueeze(1)
        # mask where (done_idx - idx) < block_size is 1 else 0
        padding_mask = torch.cat([torch.ones(done_idx - idx), torch.zeros(block_size - (done_idx - idx))], dim=0)
        return states, actions, rtgs, timesteps, padding_mask




if __name__ == "__main__":
    DATASET_ROOT = "YOUR_DATASET_PATH"
    PROCGEN_ENVS = []
    for env in os.listdir(DATASET_ROOT):
        if os.path.isdir(os.path.join(DATASET_ROOT, env)):
            PROCGEN_ENVS.append(env)

    print(PROCGEN_ENVS)
    for env in PROCGEN_ENVS:
        print(f"Processing {env}...")
        episodes_dir_path = os.path.join(DATASET_ROOT, env)
        dataset = OfflineDataset(capacity=1000000, episodes_dir_path=episodes_dir_path, percentile=1.0)

        print(env, dataset._calc_average_return())
```

./procgen/offline/evaluate_offline_agent.py
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

log_dir = os.path.expandvars(os.path.expanduser(os.path.join(args.save_path, args.env_name)))
# check if final_model.pt already exists in the log_dir
if not os.path.exists(os.path.join(log_dir, args.xpid, "final_model.pt")):
    raise FileNotFoundError("Final model does not exist in the log_dir")

if os.path.exists(os.path.join(log_dir, args.xpid, "evaluate.csv")):
    # exit if final_model.pt already exists
    print("Final evaluate csv already exists in the log_dir")
    exit(0)
    
# Load dataset
extra_config = None
if args.algo in ["dt", "bct"]:
    dataset = OfflineDTDataset(
        capacity=args.dataset_size, episodes_dir_path=os.path.join(args.dataset, args.env_name), percentile=args.percentile, context_len=args.dt_context_length, rtg_noise_prob=args.dt_rtg_noise_prob
    )
    extra_config = {"train_data_vocab_size": dataset.vocab_size, "train_data_block_size": dataset._block_size, "max_timesteps": max(dataset._timesteps), "dataset_size": len(dataset)}
    eval_max_return = dataset.get_max_return(multiplier=args.dt_eval_ret)


# create Procgen env
env = procgen.ProcgenEnv(num_envs=1, env_name=args.env_name)
env = VecExtractDictObs(env, "rgb")


# Initialize agent
agent = _create_agent(args, env=env, extra_config=extra_config)
agent.set_device(device)
print("Model Created!")

# load checkpoint and resume if resume flag is true
curr_epochs = agent.load(os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt"))
print(f"Checkpoint Loaded!")


if args.algo in ["dt", "bct"]:
    test_mean_perf = eval_DT_agent(agent, eval_max_return, device, env_name=args.env_name, start_level=args.num_levels+50, distribution_mode=args.distribution_mode, num_episodes=100)
    train_mean_perf = eval_DT_agent(
        agent, eval_max_return, device, env_name=args.env_name, num_levels=args.num_levels, start_level=0, distribution_mode=args.distribution_mode, num_episodes=100
    )
    val_mean_perf = eval_DT_agent(
        agent, eval_max_return, device, env_name=args.env_name, num_levels=50, start_level=args.num_levels, distribution_mode=args.distribution_mode, num_episodes=100
    )
else:
    test_mean_perf = eval_agent(agent, device, env_name=args.env_name, start_level=args.num_levels+50, distribution_mode=args.distribution_mode, eval_eps=args.eval_eps, num_episodes=100)
    train_mean_perf = eval_agent(
        agent, device, env_name=args.env_name, num_levels=args.num_levels, start_level=0, distribution_mode=args.distribution_mode, eval_eps=args.eval_eps, num_episodes=100
    )
    val_mean_perf = eval_agent(
        agent, device, env_name=args.env_name, num_levels=50, start_level=args.num_levels, distribution_mode=args.distribution_mode, num_episodes=100
    )

# save dict to csv in logdir
with open(os.path.join(log_dir, args.xpid, "evaluate.csv"), "w") as f:
    f.write("final_test_ret,final_train_ret,final_val_ret\n")
    f.write(f"{test_mean_perf},{train_mean_perf},{val_mean_perf}\n")
    
print(f"Final Test Return: {test_mean_perf}")
print(f"Final Train Return: {train_mean_perf}")
print(f"Final Val Return: {val_mean_perf}")

print("Done!")
```

./procgen/offline/single_level_train_offline_agent.py
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

./procgen/offline/test_offline_agent.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import procgen
import torch
import torch.nn as nn
from baselines.common.vec_env import VecExtractDictObs


def eval_agent(
    agent: nn.Module,
    device,
    env_name="miner",
    num_levels=0,
    start_level=0,
    distribution_mode="easy",
    eval_eps=0.001,
    num_episodes=10,
):
    # Sample Levels From the Full Distribution
    env = procgen.ProcgenEnv(
        num_envs=1,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
    )
    env = VecExtractDictObs(env, "rgb")

    eval_episode_rewards = []
    agent.eval()
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if obs.shape[1] != 3:
                obs = obs.transpose(0, 3, 1, 2)
            obs = torch.from_numpy(obs).float().to(device)
            # normalize obs to [0, 1] if [0,255]
            # if obs.max() > 1.0:
            #     obs /= 255.0
            action = agent.eval_step(obs, eps=eval_eps)
            # using numpy, if action is of shape [1,1] then convert it to [1]
            if action.shape == (1, 1):
                action = action[0]
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
        eval_episode_rewards.append(episode_reward)
    mean_eval_episode_reward = sum(eval_episode_rewards) / len(eval_episode_rewards)
    return mean_eval_episode_reward


def eval_DT_agent(
    agent: nn.Module,
    ret,
    device,
    env_name="miner",
    num_levels=0,
    start_level=0,
    distribution_mode="easy",
    num_episodes=10,
):
    # Sample Levels From the Full Distribution
    env = procgen.ProcgenEnv(
        num_envs=1,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
    )
    env = VecExtractDictObs(env, "rgb")

    eval_episode_rewards = []
    info_episode_rewards = []
    agent.eval()
    done = True
    for _ in range(num_episodes):
        state = env.reset()
        # done = False

        if state.shape[1] != 3:
            state = state.transpose(0, 3, 1, 2)
        state = torch.from_numpy(state).type(torch.float32).to(device).unsqueeze(0)
        # normalize state to [0, 1] if [0,255]
        if state.max() > 1.0:
            state /= 255.0

        rtgs = [ret]
        # first state is from env, first rtg is target return, and first timestep is 0
        sampled_action = agent.sample(
            state,
            1,
            temperature=1.0,
            sample=True,
            actions=None,
            rtgs=torch.tensor(rtgs, dtype=torch.float).to(device).unsqueeze(0).unsqueeze(-1),
            timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device),
        )
        j = 0
        all_states = state
        actions = []

        while True:
            if done:
                state, reward_sum, done = env.reset(), 0, False
            action = sampled_action[0]
            actions += [sampled_action]
            if action.shape == (1, 1):
                action = action[0]
            state, reward, done, infos = env.step(action.cpu().numpy())

            if state.shape[1] != 3:
                state = state.transpose(0, 3, 1, 2)
            state = torch.from_numpy(state).type(torch.float32).to(device)
            # normalize state to [0, 1] if [0,255]
            if state.max() > 1.0:
                state /= 255.0
            reward_sum += reward[0]
            j += 1

            for info in infos:
                if "episode" in info.keys():
                    info_episode_rewards.append(info["episode"]["r"])

            if done:
                eval_episode_rewards.append(reward_sum)
                break

            state = state.unsqueeze(0).to(device)

            all_states = torch.cat([all_states, state], dim=0)

            if reward.shape != (1, 1):
                reward = torch.from_numpy(reward).type(torch.float32).unsqueeze(-1)
            rtgs += [rtgs[-1] - reward]
            # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
            # timestep is just current timestep
            # st()
            sampled_action = agent.sample(
                all_states.unsqueeze(0),
                1,
                temperature=1.0,
                sample=True,
                actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0),
                rtgs=torch.tensor(rtgs, dtype=torch.float).to(device).unsqueeze(0).unsqueeze(-1),
                timesteps=(min(j, agent.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)),
            )

    mean_eval_episode_reward = sum(eval_episode_rewards) / len(eval_episode_rewards)
    # print(mean_eval_episode_reward, info_episode_rewards)
    return mean_eval_episode_reward
```

./procgen/offline/train_offline_agent.py
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

./procgen/online/__init__.py
```python
```

./procgen/online/behavior_policies/__init__.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .envs import VecPyTorchProcgen, make_venv
from .model import PPOnet
from .replay_buffer import ReplayBuffer
```

./procgen/online/behavior_policies/algos/__init__.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .ppo import PPO
```

./procgen/online/behavior_policies/algos/ppo.py
```python
# Copyright (c) 2017 Ilya Kostrikov
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim


class PPO:
    """
    PPO
    """

    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
    ):
        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ = sample

                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (values - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # Update actor-critic using both PPO Loss
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
```

./procgen/online/behavior_policies/arguments.py
```python
# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/idaac/blob/main/ppo_daac_idaac/arguments.py

import argparse

from utils.utils import str2bool

parser = argparse.ArgumentParser(description="RL")


# Algorithm arguments.
parser.add_argument("--algo", default="ppo", choices=["ppo"], help="algorithm to use")
parser.add_argument("--alpha", type=float, default=0.99, help="RMSprop optimizer apha")
parser.add_argument("--clip_param", type=float, default=0.2, help="ppo clip parameter")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="entropy term coefficient")
parser.add_argument("--eps", type=float, default=1e-5, help="RMSprop optimizer epsilon")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="gae lambda parameter")
parser.add_argument("--gamma", type=float, default=0.999, help="discount factor for rewards")
parser.add_argument("--hidden_size", type=int, default=256, help="state embedding dimension")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max norm of gradients)")
parser.add_argument("--num_mini_batch", type=int, default=8, help="number of batches for ppo")
parser.add_argument("--num_steps", type=int, default=256, help="number of forward steps in A2C")
parser.add_argument("--ppo_epoch", type=int, default=1, help="number of ppo epochs")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--value_loss_coef", type=float, default=0.5, help="value loss coefficient (default: 0.5)")

# Procgen arguments.
parser.add_argument("--distribution_mode", default="easy", help="distribution of envs for procgen")
parser.add_argument("--env_name", type=str, default="coinrun", help="environment to train on")
parser.add_argument("--num_levels", type=int, default=200, help="number of Procgen levels to use for training")
parser.add_argument("--num_processes", type=int, default=64, help="how many training CPU processes to use")
parser.add_argument("--start_level", type=int, default=0, help="start level id for sampling Procgen levels")

# Training arguments
parser.add_argument("--archive_interval", type=int, default=50, help="number of updates after which model is saved.")
parser.add_argument("--checkpoint_path", type=str, default="", help="Directory to load model to start training.")
parser.add_argument("--log_interval", type=int, default=10, help="log interval, one log per n updates")
parser.add_argument(
    "--log_wandb",
    type=str2bool,
    default=True,
    help="If true, log of parameters and gradients to wandb",
)
parser.add_argument("--no_cuda", type=str2bool, default=False, help="If true, use CPU only.")
parser.add_argument("--num_env_steps", type=int, default=25e6, help="number of environment steps to train")
parser.add_argument("--model_saving_dir", type=str, default="models", help="Directory to save model during training.")
parser.add_argument(
    "--resume", type=str2bool, default=False, help="If true, load existing checkpoint to start training."
)
parser.add_argument("--wandb_base_url", type=str, default=None, help="wandb base url")
parser.add_argument("--wandb_api_key", type=str, default=None, help="wandb api key")
parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity")
parser.add_argument("--wandb_project", type=str, default=None, help="wandb project name")
parser.add_argument("--xpid", type=str, default="debug", help="xpid name")

# Dataset arguments
parser.add_argument(
    "--dataset_saving_dir",
    type=str,
    default="dataset_saving_dir",
    help="directory to save episodes for offline training.",
)
parser.add_argument(
    "--save_dataset",
    type=str2bool,
    default=False,
    help="If true, save episodes as datasets when training behavior policy for offline training.",
)
```

./procgen/online/behavior_policies/distributions.py
```python
# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/idaac/blob/main/ppo_daac_idaac/distributions.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import init


class FixedCategorical(torch.distributions.Categorical):
    """
    Categorical distribution object
    """

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)
    
    def _get_logits(self):
        return self.logits
    
    def _get_log_softmax(self):
        log_probs = F.log_softmax(self.logits, dim=1)
        return log_probs


class Categorical(nn.Module):
    """
    Categorical distribution (NN module)
    """

    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)
```

./procgen/online/behavior_policies/envs.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
# This file is a modified version of:
# https://github.com/facebookresearch/level-replay/blob/main/level_replay/envs.py

import procgen
import torch
from baselines.common.vec_env import VecEnvWrapper, VecExtractDictObs, VecMonitor, VecNormalize
from gym.spaces.box import Box
from procgen import ProcgenEnv


class VecPyTorchProcgen(VecEnvWrapper):
    def __init__(self, venv, device, normalize=True):
        """
        Environment wrapper that returns tensors (for obs and reward)
        """
        super(VecPyTorchProcgen, self).__init__(venv)
        self.device = device
        self.normalize = normalize

        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [3, 64, 64],
            dtype=self.observation_space.dtype,
        )

    def reset(self):
        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)

        obs = torch.from_numpy(obs).float().to(self.device)
        if self.normalize:
            obs /= 255.0

        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)

        obs = torch.from_numpy(obs).float().to(self.device)
        if self.normalize:
            obs /= 255.0

        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


def make_venv(num_envs, env_name, device, **kwargs):
    """
    Function to create a vectorized environment.
    """
    if env_name in procgen.env.ENV_NAMES:
        num_levels = kwargs.get("num_levels", 1)
        start_level = kwargs.get("start_level", 0)
        distribution_mode = kwargs.get("distribution_mode", "easy")
        paint_vel_info = kwargs.get("paint_vel_info", False)
        use_sequential_levels = kwargs.get("use_sequential_levels", False)
        ret_normalization = kwargs.get("ret_normalization", False)
        obs_normalization = kwargs.get("obs_normalization", False)

        venv = ProcgenEnv(
            num_envs=num_envs,
            env_name=env_name,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
            paint_vel_info=paint_vel_info,
            use_sequential_levels=use_sequential_levels,
        )
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False, ret=ret_normalization)

        envs = VecPyTorchProcgen(venv, device, normalize=obs_normalization)
    else:
        raise ValueError(f"Unsupported env {env_name}")

    return envs
```

./procgen/online/behavior_policies/model.py
```python
# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/idaac/blob/main/ppo_daac_idaac/model.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import torch.nn as nn
import torch.nn.functional as F

from online.behavior_policies.distributions import Categorical
from utils.utils import init

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain("relu"))


def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    """
    Flatten a tensor
    """

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size)
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class NNBase(nn.Module):
    """
    Actor-Critic network (base class)
    """

    def __init__(self, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


class BasicBlock(nn.Module):
    """
    Residual Network Block
    """

    def __init__(self, n_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


class ResNetBase(NNBase):
    """
    Residual Network
    """

    def __init__(self, num_inputs, hidden_size=256, channels=[16, 32, 32]):
        super(ResNetBase, self).__init__(hidden_size)

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=stride))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        return self.critic_linear(x), x


class PolicyResNetBase(NNBase):
    """
    Residual Network
    """

    def __init__(self, num_inputs, hidden_size=256, channels=[16, 32, 32], num_actions=15):
        super(PolicyResNetBase, self).__init__(hidden_size)
        self.num_actions = num_actions

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size + num_actions, 1))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=stride))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs, actions=None):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if actions is None:
            onehot_actions = torch.zeros(x.shape[0], self.num_actions).to(x.device)
        else:
            onehot_actions = F.one_hot(actions.squeeze(1), self.num_actions).float()
        gae_inputs = torch.cat((x, onehot_actions), dim=1)

        return self.critic_linear(gae_inputs), x


class ValueResNet(NNBase):
    """
    Residual Network
    """

    def __init__(self, num_inputs, hidden_size=256, channels=[16, 32, 32]):
        super(ValueResNet, self).__init__(hidden_size)

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=stride))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        return self.critic_linear(x)


class LinearOrderClassifier(nn.Module):
    def __init__(self, emb_size=256):
        super(LinearOrderClassifier, self).__init__()
        self.main = nn.Sequential(
            Flatten(),
            init_(nn.Linear(2 * emb_size, 2)),
            nn.Softmax(dim=1),
        )
        self.train()

    def forward(self, emb):
        x = self.main(emb)
        return x


class NonlinearOrderClassifier(nn.Module):
    def __init__(self, emb_size=256, hidden_size=4):
        super(NonlinearOrderClassifier, self).__init__()
        self.main = nn.Sequential(
            Flatten(),
            init_relu_(nn.Linear(2 * emb_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, 2)),
            nn.Softmax(dim=1),
        )
        self.train()

    def forward(self, emb):
        x = self.main(emb)
        return x


class PPOnet(nn.Module):
    """
    PPO netowrk
    """

    def __init__(self, obs_shape, num_actions, base_kwargs=None):
        super(PPOnet, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}

        base = ResNetBase

        self.base = base(obs_shape[0], **base_kwargs)
        self.dist = Categorical(self.base.output_size, num_actions)

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
```

./procgen/online/behavior_policies/replay_buffer.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import datetime
import io
import os
import tempfile
from typing import Dict

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def save_episode(episode: Dict[str, np.ndarray], directory: str, filename: str):
    os.makedirs(directory, exist_ok=True)
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with open(os.path.join(directory, filename), "wb") as f:
            f.write(bs.read())


class ReplayBuffer(object):
    """
    Create a replay buffer for procgen environment.

    Procgen Environment information: https://github.com/openai/procgen#environments
    """

    def __init__(
        self,
        observation_space,
        action_space,
        n_envs: int = 1,
        n_steps: int = 256,
        save_episode: bool = False,
        storage_path: str = None,
    ) -> None:
        """

        Args:
            observation_space (gym.spaces.Box): the observation space for the environment
            action_space (gym.spaces.Discrete): the action space for the environment
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.n_steps = n_steps

        self.n_episodes = 0
        self.save_episode = save_episode
        self.storage_path = storage_path
        if save_episode and storage_path is None:
            self.storage_path = tempfile.mkdtemp(prefix="replay_buffer_")

        self.setup()

    def setup(self) -> None:
        """
        Initializing buffers and indexes
        """
        # Buffers for each element
        self._obs_buffer = torch.zeros(size=(self.n_steps + 1, self.n_envs, *self.observation_space))
        self._reward_buffer = torch.zeros(size=(self.n_steps, self.n_envs, 1))
        self._done_buffer = torch.empty(size=(self.n_steps, self.n_envs), dtype=torch.bool)
        self.value_preds = torch.zeros(size=(self.n_steps + 1, self.n_envs, 1))
        self.returns = torch.zeros(size=(self.n_steps + 1, self.n_envs, 1))
        self._action_log_probs = torch.zeros(size=(self.n_steps, self.n_envs, 1))

        action_shape = 1 if self._is_discrete() else self.action_space.shape[0]
        self._action_buffer = torch.zeros(size=(self.n_steps, self.n_envs, action_shape))
        if self._is_discrete():
            self._action_buffer = self._action_buffer.long()

        self._masks = torch.ones(size=(self.n_steps + 1, self.n_envs, 1))

        # Index
        self._idx = 0
        self._prev_idx = -1

    def insert_initial_observations(self, init_obs) -> None:
        """
        Add only the first observations obtained when reseting the environments.

        Example:
            >>> venv = ProcgenEnv(num_envs=3, env_name="miner", num_levels=200, start_level=0, distribution_mode="easy")
            >>> venv = VecExtractDictObs(venv, "rgb")
            >>> obs = venv.reset()
            >>> buffer = ReplayBuffer(
                    observation_space=venv.observation_space,
                    action_space=venv.action_space,
                    n_envs=venv.num_envs,
                    n_steps=4,
                    save_episode=False,
                )
            >>> buffer.setup()

            >>> buffer.insert_initial_observations(obs)

        Args:
            init_obs (np.ndarray): initial observations
        """
        self._obs_buffer[self._idx].copy_(init_obs)

        if self.save_episode:
            if not hasattr(self, "ongoing_episodes"):
                self.ongoing_episodes = [None] * self.n_envs

            for env_idx in range(self.n_envs):
                self.ongoing_episodes[env_idx] = collections.defaultdict(list)
                self.ongoing_episodes[env_idx]["obs"].append(
                    np.array(init_obs[env_idx].detach().cpu())
                )  # init_obs: (n_envs, 3, 64, 64)

    def to(self, device):
        """
        Move tensors in the buffers to a CPU / GPU device.
        """
        self._obs_buffer = self._obs_buffer.to(device)
        self._reward_buffer = self._reward_buffer.to(device)
        self._done_buffer = self._done_buffer.to(device)
        self._action_buffer = self._action_buffer.to(device)
        self._action_log_probs = self._action_log_probs.to(device)
        self.returns = self.returns.to(device)
        self.value_preds = self.value_preds.to(device)
        self._masks = self._masks.to(device)

    def insert(self, obs, actions, rewards, dones, action_log_probs, value_preds, masks) -> None:
        r"""
        Insert tuple of (observations, actions, rewards, dones) into corresponding buffers.
        An 's' at the end of each parameter indicates that the environments can be vectorized.

        Example:
            >>> venv = ProcgenEnv(num_envs=3, env_name="miner", num_levels=200, start_level=0, distribution_mode="easy")
            >>> venv = VecExtractDictObs(venv, "rgb")
            >>> obs = venv.reset()

            >>> action = venv.action_space.sample()
            >>> actions = np.array([action] * venv.num_envs)
            >>> obs, rewards, dones, _infos = venv.step(actions)

            >>> buffer.insert(obs, actions, rewards, dones)

        Args:
            obs: observations or states observed from the environments after the agents perform the actions.
            actions: The actions sampled from a certain policy or randomly for the agents to perform.
            rewards: The immidiate rewards that the agents received from the environments after performing the actions.
            dones: A boolean vector whose elements indicate whether the episode terminates in an environment.
        """
        if len(rewards.shape) == 3:
            rewards = rewards.squeeze(2)

        self._obs_buffer[self._idx + 1].copy_(obs)
        self._action_buffer[self._idx].copy_(actions)
        self._reward_buffer[self._idx].copy_(rewards)
        self._done_buffer[self._idx] = torch.tensor(dones)
        self._action_log_probs[self._idx].copy_(action_log_probs)
        self.value_preds[self._idx].copy_(value_preds)
        self._masks[self._idx + 1].copy_(masks)

        # Update index
        self._prev_idx = self._idx
        self._idx = (self._idx + 1) % self.n_steps

        if self.save_episode:
            for env_idx in range(self.n_envs):
                self.ongoing_episodes[env_idx]["obs"].append(np.array(obs[env_idx].detach().cpu()))
                self.ongoing_episodes[env_idx]["actions"].append(np.array(actions[env_idx].detach().cpu()))
                self.ongoing_episodes[env_idx]["rewards"].append(np.array(rewards[env_idx].detach().cpu()))
                self.ongoing_episodes[env_idx]["dones"].append(dones[env_idx])

            if any(dones):
                done_env_idxs = np.where(dones == 1)[0]
                self._save_terminated_episodes(done_env_idxs)
                self._reset_terminated_episodes(done_env_idxs)

    def after_update(self):
        self._obs_buffer[0].copy_(self._obs_buffer[-1])
        self._masks[0].copy_(self._masks[-1])

    def compute_returns(self, next_value, gamma, gae_lambda):
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self._reward_buffer.shape[0])):
            delta = (
                self._reward_buffer[step]
                + gamma * self.value_preds[step + 1] * self._masks[step + 1]
                - self.value_preds[step]
            )
            gae = delta + gamma * gae_lambda * self._masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        n_steps, n_envs = self._reward_buffer.shape[0:2]
        batch_size = n_envs * n_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                f"PPO requires the number of processes ({n_envs}) ",
                f"* number of steps ({n_steps}) = {n_envs * n_steps} ",
                f"to be greater than or equal to the number of PPO mini batches ({num_mini_batch}).",
            )
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

        for indices in sampler:
            obs_batch = self._obs_buffer[:-1].view(-1, *self._obs_buffer.shape[2:])[indices]
            actions_batch = self._action_buffer.view(-1, self._action_buffer.shape[-1])[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self._action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, actions_batch, value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ

    def save(self) -> None:
        """
        Save the replay buffer
        """
        pass

    def _is_discrete(self) -> bool:
        """
        Determine if the environment action space is discrete or continuous
        """
        return self.action_space.__class__.__name__ == "Discrete"

    def _save_terminated_episodes(self, done_env_idxs: np.ndarray) -> None:
        """
        Save all terminated episodes among the n_envs environments.

        Args:
            done_env_idxs (np.ndarray): indexs of environments having episode terminated at the current step.
        """
        for env_idx in done_env_idxs:
            self._save_episode(env_idx)

    def _save_episode(self, env_idx: int) -> None:
        """
        Save a single episode into file.

        Args:
            env_idx (int): the index of the environment, ranging [0, n_envs-1] inclusive.
        """
        # Convert list to numpy array
        episode_idx = self.n_episodes
        episode_len = len(self.ongoing_episodes[env_idx]["rewards"])
        episode = {}
        for k, v in self.ongoing_episodes[env_idx].items():
            first_value = v[0]
            if isinstance(first_value, np.ndarray):
                dtype = first_value.dtype
            elif isinstance(first_value, int):
                dtype = np.int64
            elif isinstance(first_value, float):
                dtype = np.float32
            elif isinstance(first_value, bool):
                dtype = np.bool_
            episode[k] = np.array(v, dtype=dtype)

        # Store the episode
        self.n_episodes += 1
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        episode_filename = f"{timestamp}_{episode_idx}_{episode_len}.npz"
        save_episode(episode, self.storage_path, episode_filename)

    def _reset_terminated_episodes(self, done_env_idxs: np.ndarray) -> None:
        """
        Reset references of all terminated episodes of the n_envs environments.

        Args:
            done_env_idxs (np.ndarray): indexs of environments having episode terminated at the current step.
        """
        for env_idx in done_env_idxs:
            self._reset_terminated_episode(env_idx)

    def _reset_terminated_episode(self, env_idx: int) -> None:
        """
        Reset the reference of a single terminated episode.

        Args:
            env_idx (int): the index of the environment, ranging [0, n_envs-1] inclusive.
        """
        # clear the reference
        self.ongoing_episodes[env_idx] = collections.defaultdict(list)

        # the next_obs of the previous (saved) terminated episode is the init_obs of the next episode.
        # self._prev_idx has range [1, n_steps+1] inclusive, covers the roll over edge cases.
        self.ongoing_episodes[env_idx]["obs"].append(
            np.array(self._obs_buffer[self._prev_idx + 1][env_idx].detach().cpu())
        )
```

./procgen/online/data_collector.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib
from typing import Generator

import numpy as np
import torch
import torch.nn as nn

import baselines
from online.behavior_policies import PPOnet, make_venv
from online.datasets import RolloutStorage, arguments
from utils.utils import LogDirType, LogItemType, set_seed

DEFAULT_CHECKPOINT_DIRECTORY = "YOUR_PPO_CHECKPOINT_DIR"


def collect_data(args):
    print("\nArguments: ", args)
    if args.ratio is not None:
        assert args.ratio >= 0.0 and args.ratio <= 1.0, "The ratio should be between 0 and 1!"

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
            "ret_normalization": False,
            "obs_normalization": False,
        },
    )
    envs.reset()

    # Initialize Model and Replay Buffer
    obs_shape = envs.observation_space.shape
    model = PPOnet(obs_shape, envs.action_space.n, base_kwargs={"hidden_size": args.hidden_size})

    rollouts = RolloutStorage(
        envs.observation_space.shape,
        action_space=envs.action_space,
        n_envs=args.num_processes,
        capacity=args.capacity,
        save_episode=args.save_dataset,
        storage_path=args.dataset_saving_dir,
    )
    if args.ratio is None:
        _roll_out(model, args.seed, envs, rollouts, device, args.checkpoint_path, args.num_env_steps)
    else:
        env_dir = os.path.join(
            DEFAULT_CHECKPOINT_DIRECTORY if args.ratio_checkpoint_dir is None else args.ratio_checkpoint_dir,
            args.env_name,
        )  # env_dir = '/checkpoint/offlinerl/ppo/miner'

        # Sub_env_dirs be like:
        # ['/checkpoint/offlinerl/ppo/miner/xpid_0/', '/checkpoint/offlinerl/ppo/miner/xpid_1/', ...]
        sub_env_dirs = [f.path for f in os.scandir(env_dir) if f.is_dir()]

        for sub_env_dir in sub_env_dirs:
            seed = int(sub_env_dir[-1])
            # Storage path be like: '/checkpoint/offlinerl/ppo/miner/xpid_0/dataset/0.75/'
            rollouts.set_storage_path(os.path.join(sub_env_dir, args.ratio_dataset_dir, str(args.ratio)))
            checkpoint_path = _pick_checkpoint_from_pool(args.ratio, directory=sub_env_dir)
            _roll_out(model, seed, envs, rollouts, device, checkpoint_path, args.num_env_steps)


def _roll_out(
    model: nn.Module,
    seed: int,
    envs: baselines.common.vec_env.VecEnvWrapper,
    rollouts: RolloutStorage,
    device: torch.device,
    checkpoint_path: str,
    target_env_steps: int,
):
    set_seed(seed)

    # Fetch current observations from environment
    obs, _, _, _ = envs.step_wait()
    rollouts.reset()
    rollouts.insert_initial_observations(obs)
    rollouts.to(device)

    # Load checkpoint
    _load_checkpoint(model, checkpoint_path)
    model.to(device)
    print("\n Neural Network: ", model)

    # Roll out the agent and collect Data
    model.eval()
    prev_done_indexes = np.zeros(shape=(envs.num_envs,), dtype=int)
    saved_env_steps = np.zeros(shape=(envs.num_envs,), dtype=int)
    step = 0
    while sum(saved_env_steps) <= target_env_steps:
        # Sample actions
        with torch.no_grad():
            # Raw obs will be saved in rollout storage, while the model needs normalized obs
            # since it was trained with normalized obs.
            _value, action, _action_log_prob = model.act(obs / 255.0)

        # Move one step forward
        obs, rewards, dones, infos = envs.step(action)

        # Store results
        rollouts.insert(obs, action, rewards, dones, infos)

        # Calculate saved env steps
        if any(dones):
            done_env_idxs = np.where(dones == 1)[0]
            saved_env_steps[done_env_idxs] += step - prev_done_indexes[done_env_idxs]
            prev_done_indexes[done_env_idxs] = step

        step += 1


def _load_checkpoint(model: nn.Module, path: str):
    checkpoint_path = os.path.expandvars(os.path.expanduser(f"{path}"))

    print(f"Loading checkpoint from {checkpoint_path} ...")
    try:
        checkpoint_states = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_states[LogItemType.MODEL_STATE_DICT.value])
    except Exception:
        print(f"Unable to load checkpoint from {checkpoint_path}, model is initialized randomly.")


def _pick_checkpoint_from_pool(ratio: float, directory: str) -> pathlib.Path:
    """
    Choose a checkpoint whose expected test return is of ratio (between 0 and 1)
    of that of an expert checkpoint.
    """
    final_checkpoint_path = next(_fetch_final_checkpoint_paths(directory))
    if ratio == 1.0:
        return final_checkpoint_path

    # Fetch all checkpoints with expected test returns
    names = [p for p in _fetch_checkpointed_checkpoint_paths(directory)]
    scores = (_fetch_return_from_name(name) for name in names)
    path_to_score = dict((name, score) for name, score in zip(names, scores))

    # Find the checkpoint closest to the target return.
    target_return = ratio * _fetch_return_from_name(final_checkpoint_path)
    closest_checkpoint = sorted(path_to_score.items(), key=lambda pair: abs(pair[1] - target_return))[0][0]
    return closest_checkpoint


def _fetch_final_checkpoint_paths(directory: str) -> Generator[pathlib.Path, None, None]:
    return _fetch_checkpoint_paths(directory, LogDirType.FINAL)


def _fetch_checkpointed_checkpoint_paths(directory: str) -> Generator[pathlib.Path, None, None]:
    return _fetch_checkpoint_paths(directory, LogDirType.CHECKPOINT)


def _fetch_checkpoint_paths(directory: str, dir_type: LogDirType) -> Generator[pathlib.Path, None, None]:
    return (n for n in pathlib.Path(os.path.join(directory, dir_type.value)).rglob("*.pt"))


def _fetch_return_from_name(name: pathlib.Path) -> float:
    """
    Example:
    >>> name = pathlib.Path('/my_documents/model_1_3.5.pt')
    >>> score = fetch_return_from_name(name)
    >>> score
    3.5
    """
    return float(name.stem.split("_")[-1])


if __name__ == "__main__":
    args = arguments.parser.parse_args()

    collect_data(args)

    print("Data collection completed!")
```

./procgen/online/datasets/__init__.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .storage import RolloutStorage
```

./procgen/online/datasets/arguments.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from utils.utils import str2bool

parser = argparse.ArgumentParser(description="Data Collection")

# Checkpoint arguments
checkpoints = parser.add_mutually_exclusive_group()
checkpoints.add_argument("--checkpoint_path", type=str, help="path to the model checkpoint")
checkpoints.add_argument(
    "--ratio",
    type=float,
    default=None,
    metavar="[0.0 - 1.0]",
    help="Ratio of the expected test return of a checkpoint compared to expert data.",
)
parser.add_argument(
    "--ratio_checkpoint_dir", type=str, help="Directory where search of targeted checkpoint will happen."
)
parser.add_argument(
    "--ratio_dataset_dir",
    type=str,
    default="dataset",
    help="Root directory name for collected data under each checkpoint.",
)

# Dataset arguments
parser.add_argument(
    "--capacity",
    type=int,
    default=int(1e3),
    help="Size of the table to store interactions of the agent with the environment.",
)
parser.add_argument(
    "--dataset_saving_dir",
    type=str,
    default="dataset_saving_dir",
    help="directory to save episodes for offline training.",
)
parser.add_argument("--gae_lambda", type=float, default=0.95, help="gae lambda parameter")
parser.add_argument("--gamma", type=float, default=0.999, help="discount factor for rewards")
parser.add_argument("--hidden_size", type=int, default=256, help="state embedding dimension")
parser.add_argument("--no_cuda", type=str2bool, default=False, help="If true, disable CUDA.")
parser.add_argument(
    "--num_env_steps",
    type=int,
    default=1e6,
    help="number of environment steps for the agent to interact with the environment.",
)
parser.add_argument("--num_processes", type=int, default=64, help="how many training CPU processes to use")
parser.add_argument(
    "--save_dataset",
    type=str2bool,
    help="If true, save episodes as datasets for offline training when training behavior policy.",
)
parser.add_argument("--seed", type=int, default=0, help="random seed")

# Procgen arguments.
parser.add_argument(
    "--distribution_mode",
    default="easy",
    choices=["easy", "hard", "extreme", "memory", "exploration"],
    help="distribution of envs for procgen",
)
parser.add_argument("--env_name", type=str, default="coinrun", help="environment to train on")
parser.add_argument("--num_levels", type=int, default=200, help="number of Procgen levels to use for training")
parser.add_argument("--start_level", type=int, default=0, help="start level id for sampling Procgen levels")
```

./procgen/online/datasets/storage.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import datetime
import io
import os
import tempfile
from typing import Dict

import numpy as np
import torch

from utils.utils import DatasetItemType


def save_episode(episode: Dict[str, np.ndarray], path: str):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with open(path, "wb") as f:
            f.write(bs.read())


class RolloutStorage(object):
    def __init__(
        self,
        observation_space,
        action_space,
        n_envs: int = 1,
        capacity: int = int(1e6),
        save_episode: bool = False,
        storage_path: str = None,
    ) -> None:
        """

        Args:
            observation_space (gym.spaces.Box): the observation space for the environment
            action_space (gym.spaces.Discrete): the action space for the environment
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_envs = n_envs
        self.capacity = capacity

        self.n_episodes = 0
        self.save_episode = save_episode
        self.storage_path = storage_path
        self.set_storage_path(storage_path)

        self.setup()

    def setup(self) -> None:
        """
        Initializing buffers and indexes
        """
        # Buffers for each element
        self._obs_buffer = torch.zeros(size=(self.capacity + 1, self.n_envs, *self.observation_space))
        self._reward_buffer = torch.zeros(size=(self.capacity, self.n_envs, 1))
        self._done_buffer = torch.empty(size=(self.capacity, self.n_envs), dtype=torch.bool)

        action_shape = 1 if self._is_discrete() else self.action_space.shape[0]
        self._action_buffer = torch.zeros(size=(self.capacity, self.n_envs, action_shape))
        if self._is_discrete():
            self._action_buffer = self._action_buffer.long()

        self._level_seeds_buffer = np.zeros(shape=(self.n_envs,), dtype=np.int32)

        # Index
        self._idx = 0
        self._prev_idx = -1

    def reset(self) -> None:
        """
        Reset buffers and indexes
        """
        # Buffers for each element
        self._obs_buffer.zero_()
        self._reward_buffer.zero_()
        self._done_buffer.zero_()
        self._action_buffer.zero_()
        self._level_seeds_buffer = np.zeros(shape=(self.n_envs,), dtype=np.int32)

        # Index
        self._idx = 0
        self._prev_idx = -1

    def set_storage_path(self, storage_path) -> None:
        if not self.save_episode:
            return

        self.storage_path = storage_path
        if not os.path.exists(storage_path):
            os.makedirs(storage_path, exist_ok=True)

        if storage_path is None:
            self.storage_path = tempfile.mkdtemp(prefix="replay_buffer_")

    def insert_initial_observations(self, init_obs) -> None:
        """
        Add only the first observations obtained when reseting the environments.

        Example:
            >>> venv = ProcgenEnv(num_envs=3, env_name="miner", num_levels=200, start_level=0, distribution_mode="easy")
            >>> venv = VecExtractDictObs(venv, "rgb")
            >>> obs = venv.reset()
            >>> buffer = ReplayBuffer(
                    observation_space=venv.observation_space,
                    action_space=venv.action_space,
                    n_envs=venv.num_envs,
                    capacity=4,
                    save_episode=False,
                )
            >>> buffer.setup()

            >>> buffer.insert_initial_observations(obs)

        Args:
            init_obs (np.ndarray): initial observations
        """
        self._obs_buffer[self._idx].copy_(init_obs)

        if self.save_episode:
            if not hasattr(self, "ongoing_episodes"):
                self.ongoing_episodes = [None] * self.n_envs

            for env_idx in range(self.n_envs):
                self.ongoing_episodes[env_idx] = collections.defaultdict(list)
                self.ongoing_episodes[env_idx][DatasetItemType.OBSERVATIONS.value].append(
                    init_obs[env_idx].detach().cpu().numpy()
                )  # init_obs: (n_envs, 3, 64, 64)

    def to(self, device):
        """
        Move tensors in the buffers to a CPU / GPU device.
        """
        self._obs_buffer = self._obs_buffer.to(device)
        self._reward_buffer = self._reward_buffer.to(device)
        self._done_buffer = self._done_buffer.to(device)
        self._action_buffer = self._action_buffer.to(device)

    def insert(self, obs, actions, rewards, dones, infos=[]) -> None:
        r"""
        Insert tuple of (observations, actions, rewards, dones) into corresponding buffers.
        An 's' at the end of each parameter indicates that the environments can be vectorized.

        Example:
            >>> venv = ProcgenEnv(num_envs=3, env_name="miner", num_levels=200, start_level=0, distribution_mode="easy")
            >>> venv = VecExtractDictObs(venv, "rgb")
            >>> obs = venv.reset()

            >>> action = venv.action_space.sample()
            >>> actions = np.array([action] * venv.num_envs)
            >>> obs, rewards, dones, _infos = venv.step(actions)

            >>> buffer.insert(obs, actions, rewards, dones)

        Args:
            obs: observations or states observed from the environments after the agents perform the actions.
            actions: The actions sampled from a certain policy or randomly for the agents to perform.
            rewards: The immidiate rewards that the agents received from the environments after performing the actions.
            dones: A boolean vector whose elements indicate whether the episode terminates in an environment.
        """
        # Update buffers
        if len(rewards.shape) == 3:
            rewards = rewards.squeeze(2)

        self._obs_buffer[self._idx + 1].copy_(obs)
        self._action_buffer[self._idx].copy_(actions)
        self._reward_buffer[self._idx].copy_(rewards)
        self._done_buffer[self._idx] = torch.tensor(dones)
        # When done == True (episode is completed), 'level_seed' in info will be the new seed,
        # and 'prev_level_seed' will be the seed that was used in that completed episode.
        # Save 'prev_level_seed' before saving an completed episode.
        self._level_seeds_buffer = np.array([info.get("prev_level_seed", -1) for info in infos])

        # Update index
        self._prev_idx = self._idx
        self._idx = (self._idx + 1) % self.capacity

        # Keep track of current episodes and save completed ones.
        if self.save_episode:
            for env_idx in range(self.n_envs):
                self.ongoing_episodes[env_idx][DatasetItemType.OBSERVATIONS.value].append(
                    obs[env_idx].detach().cpu().numpy()
                )
                self.ongoing_episodes[env_idx][DatasetItemType.ACTIONS.value].append(
                    actions[env_idx].detach().cpu().numpy()
                )
                self.ongoing_episodes[env_idx][DatasetItemType.REWARDS.value].append(
                    rewards[env_idx].detach().cpu().numpy()
                )
                self.ongoing_episodes[env_idx][DatasetItemType.DONES.value].append(dones[env_idx])

            if any(dones):
                done_env_idxs = np.where(dones == 1)[0]
                self._save_terminated_episodes(done_env_idxs)
                self._reset_terminated_episodes(done_env_idxs)

        # Update with current level seed after saving completed episodes.
        self._level_seeds_buffer = np.array([info.get("level_seed", -1) for info in infos])

    def save(self) -> None:
        """
        Save the replay buffer
        """
        pass

    def _is_discrete(self) -> bool:
        """
        Determine if the environment action space is discrete or continuous
        """
        return self.action_space.__class__.__name__ == "Discrete"

    def _save_terminated_episodes(self, done_env_idxs: np.ndarray) -> None:
        """
        Save all terminated episodes among the n_envs environments.

        Args:
            done_env_idxs (np.ndarray): indexs of environments having episode terminated at the current step.
        """
        for env_idx in done_env_idxs:
            self._save_episode(env_idx)

    def _save_episode(self, env_idx: int) -> None:
        """
        Save a single episode into file.

        Args:
            env_idx (int): the index of the environment, ranging [0, n_envs-1] inclusive.
        """
        # Convert list to numpy array
        episode = {}
        for k, v in self.ongoing_episodes[env_idx].items():
            first_value = v[0]
            if isinstance(first_value, np.ndarray):
                dtype = first_value.dtype
            elif isinstance(first_value, int):
                dtype = np.int64
            elif isinstance(first_value, float):
                dtype = np.float32
            elif isinstance(first_value, bool):
                dtype = np.bool_
            episode[k] = np.array(v, dtype=dtype)

        # File name
        episode_idx = self.n_episodes
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        episode_len = len(self.ongoing_episodes[env_idx]["rewards"])
        level_seed = self._level_seeds_buffer[env_idx]
        total_rewards = np.squeeze(sum(self.ongoing_episodes[env_idx]["rewards"]))
        episode_filename = f"{timestamp}_{episode_idx}_{episode_len}_{level_seed}_{total_rewards:.2f}.npz"

        # Store the episode
        self.n_episodes += 1
        save_episode(episode, os.path.join(self.storage_path, episode_filename))

    def _reset_terminated_episodes(self, done_env_idxs: np.ndarray) -> None:
        """
        Reset references of all terminated episodes of the n_envs environments.

        Args:
            done_env_idxs (np.ndarray): indexs of environments having episode terminated at the current step.
        """
        for env_idx in done_env_idxs:
            self._reset_terminated_episode(env_idx)

    def _reset_terminated_episode(self, env_idx: int) -> None:
        """
        Reset the reference of a single terminated episode.

        Args:
            env_idx (int): the index of the environment, ranging [0, n_envs-1] inclusive.
        """
        # clear the reference
        self.ongoing_episodes[env_idx] = collections.defaultdict(list)

        # the next_obs of the previous (saved) terminated episode is the init_obs of the next episode.
        # self._prev_idx has range [1, capacity+1] inclusive, covers the roll over edge cases.
        self.ongoing_episodes[env_idx][DatasetItemType.OBSERVATIONS.value].append(
            self._obs_buffer[self._prev_idx + 1][env_idx].detach().cpu().numpy()
        )
```

./procgen/online/evaluation.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from online.behavior_policies.envs import make_venv


def evaluate(args, model: nn.Module, device, num_episodes=10):
    model.eval()

    # Sample Levels From the Full Distribution
    eval_envs = make_venv(
        num_envs=1,
        env_name=args.env_name,
        device=device,
        **{
            "num_levels": 200,
            "start_level": 0,
            "distribution_mode": args.distribution_mode,
            "ret_normalization": False,
            "obs_normalization": True,
        },
    )

    eval_episode_rewards = []
    obs = eval_envs.reset()

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            _, action, _ = model.act(obs)

        obs, _reward, _done, infos = eval_envs.step(action)

        for info in infos:
            if "episode" in info.keys():
                eval_episode_rewards.append(info["episode"]["r"])

    eval_envs.close()
    return eval_episode_rewards
```

./procgen/online/get_ppo.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Run PPO final checkpoints on Procgen test levels
import argparse
import os
import random

import numpy as np
import torch

from online.behavior_policies.envs import make_venv
from online.behavior_policies.model import PPOnet
from utils.utils import LogItemType, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--xpid", type=str, default="debug")


def evaluate_one_game(env_name: str, num_episodes=5, seed=1, start_level=0, num_levels=0):
    # Initialize model
    model = PPOnet((3, 64, 64), 15, base_kwargs={"hidden_size": 256})
    # Load checkpoint
    # SEED = 1
    model_path = "YOUR_MODEL_PATH"
    model_path = [f.path for f in os.scandir(model_path)][0]
    print(f"Loading checkpoint from {model_path} ...")
    try:
        checkpoint_states = torch.load(model_path)
        model.load_state_dict(checkpoint_states[LogItemType.MODEL_STATE_DICT.value])
    except Exception:
        print(f"Unable to load checkpoint from {model_path}, model is initialized randomly.")
    device = "cuda:0"
    model.to(device)
    # Initialize Env

    test_envs = make_venv(
        num_envs=1,
        env_name=env_name,
        device=device,
        **{
            "num_levels": num_levels,
            "start_level": start_level,
            "distribution_mode": "easy",
            "ret_normalization": False,
            "obs_normalization": True,
        },
    )
    # # Roll out
    # model.eval()
    # rewards_per_level = []
    # obs = test_envs.reset()
    # while len(rewards_per_level) < num_episodes:
    #     with torch.no_grad():
    #         _, action, _ = model.act(obs)
    #     obs, _reward, _done, infos = test_envs.step(action)
    #     for info in infos:
    #         if "episode" in info.keys():
    #             rewards_per_level.append(info["episode"]["r"])
    # # print(f"LEVEL: {level} - AVERAGE REWARDS OVER {num_episodes} EPISODES: {np.mean(rewards_per_level)}")
    # rewards_over_all_levels.append(np.mean(rewards_per_level))
    
    eval_episode_rewards = []
    model.eval()
    for _ in range(num_episodes):
        obs = test_envs.reset()
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                _, action, _ = model.act(obs)
            obs, reward, done, _ = test_envs.step(action)
            episode_reward += reward.item()
        eval_episode_rewards.append(episode_reward)
    return eval_episode_rewards

from procgen.env import ENV_NAMES

# random.seed(1337)

args = parser.parse_args()
set_seed(args.seed)
# Change test levels based on the num_levels that PPO is trained!!!
# levels = random.sample(range(100050, 10_000_000), 10)
# print(levels)
train_result = {}
for env_name in ENV_NAMES:
    # env_name = "plunder"
    rewards = evaluate_one_game(env_name, num_episodes=10, seed=args.seed, start_level=40, num_levels=1)
    print(f"ENV: {env_name} - REWARDS OVER LEVELS: {np.mean(rewards)}")
    train_result[env_name] = np.mean(rewards)
print(train_result)

# test_result = {}
# for env_name in ENV_NAMES:
#     # env_name = "plunder"
#     rewards = evaluate_one_game(env_name, num_episodes=100, seed=args.seed, start_level=250, num_levels=0)
#     print(f"ENV: {env_name} - REWARDS OVER LEVELS: {np.mean(rewards)}")
#     test_result[env_name] = np.mean(rewards)
# print(test_result)

```

./procgen/online/trainer.py
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

./procgen/train_scripts/__init__.py
```python
```

./procgen/train_scripts/arguments.py
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

./procgen/train_scripts/cmd_generator.py
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

./procgen/train_scripts/make_cmd.py
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

./procgen/train_scripts/slurm.py
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

./procgen/utils/__init__.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .archs import DQNEncoder, BCQEncoder, PPOResNetBaseEncoder, PPOResNet20Encoder, BCQResnetBaseEncoder

AGENT_CLASSES = {
    "dqn": DQNEncoder,
    "bcq": BCQEncoder,
    "pporesnetbase": PPOResNetBaseEncoder,
    "pporesnet20": PPOResNet20Encoder,
    "bcqresnetbase": BCQResnetBaseEncoder,
}
```

./procgen/utils/archs.py
```python
# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/idaac/blob/main/ppo_daac_idaac/model.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import init

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain("relu"))

def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DQNEncoder(nn.Module):
    # Implements DQN 3-layer CNN encoder
    def __init__(self, observation_space, action_space=15, hidden_size=64, use_actor_linear=True):
        super(DQNEncoder, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(observation_space.shape[-1], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, hidden_size, 4, stride=2)
        self.conv3 = nn.Conv2d(hidden_size, 64, 3, stride=1)
        self.fc = nn.Linear(1024, action_space)
        
        apply_init_(self.modules())

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class IQLCritic(nn.Module):
    # implements IQL critic network
    def __init__(self, base_class, observation_space, action_space=15, hidden_size=64):
        super(IQLCritic, self).__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size

        self.critic_feats = 256

        self.critic_linear = nn.Linear(action_space + self.critic_feats, 1)
        self.base = PPOResNetBaseEncoder(observation_space, self.critic_feats, hidden_size)
        
        apply_init_(self.modules())

    def forward(self, x, a):
        """_summary_

        Args:
            x: observation of shape (batch_size, *observation_space.shape)
            a: actions of shape (batch_size, 1)

        Returns:
            value of shape (batch_size, 1)
        """
        x = self.base(x)  # (batch_size, critic_feats)
        q_val = self.critic_linear(torch.cat([x, a], dim=1))  # (batch_size, 1)
        return q_val


class BCQEncoder(nn.Module):
    # Source: https://github.com/sfujim/BCQ/blob/master/discrete_BCQ/discrete_BCQ.py#LL8-L31C48
    def __init__(self, observation_space, action_space=15, hidden_size=64):
        super(BCQEncoder, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(observation_space.shape[-1], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, hidden_size, 4, stride=2)
        self.conv3 = nn.Conv2d(hidden_size, 64, 3, stride=1)

        self.q1 = nn.Linear(1024, 512)
        self.q2 = nn.Linear(512, action_space)

        self.i1 = nn.Linear(1024, 512)
        self.i2 = nn.Linear(512, action_space)
        
        apply_init_(self.modules())

    def forward(self, x):
        c = F.relu(self.conv1(x))
        c = F.relu(self.conv2(c))
        c = F.relu(self.conv3(c))

        q = F.relu(self.q1(c.reshape(-1, 1024)))
        i = F.relu(self.i1(c.reshape(-1, 1024)))
        i = self.i2(i)
        return self.q2(q), F.log_softmax(i, dim=1), i


class Flatten(nn.Module):
    """
    Flatten a tensor
    """

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size)
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class BasicBlock(nn.Module):
    """
    Residual Network Block
    """

    def __init__(self, n_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


class NNBase(nn.Module):
    """
    Actor-Critic network (base class)
    """

    def __init__(self, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


class PPOResNetBaseEncoder(NNBase):
    """
    Residual Network from PPO implementation -> 1M parameters
    """

    def __init__(self, observation_space, action_space=15, hidden_size=256, channels=[16, 32, 32], use_actor_linear=True):
        super(PPOResNetBaseEncoder, self).__init__(hidden_size)
        self.observation_space = observation_space
        self.use_actor_linear = use_actor_linear

        self.layer1 = self._make_layer(observation_space.shape[-1], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        if self.use_actor_linear:
            self.actor_linear = init_(nn.Linear(hidden_size, action_space))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.use_actor_linear:
            return self.actor_linear(x)
        
        return x


class PPOResNet20Encoder(NNBase):
    """
    Residual Network with 20 layers -> 35M parameters
    """

    def __init__(self, observation_space, action_space=15, hidden_size=256, channels=[64, 256, 256, 512], use_actor_linear=True):
        super(PPOResNet20Encoder, self).__init__(hidden_size)
        self.observation_space = observation_space
        self.use_actor_linear = use_actor_linear
        
        self.layer1 = self._make_layer(observation_space.shape[-1], channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])
        self.layer4 = self._make_layer(channels[2], channels[3])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc1 = init_relu_(nn.Linear(8192, 2048))
        self.fc2 = init_relu_(nn.Linear(2048, hidden_size))

        if self.use_actor_linear:
            self.actor_linear = init_(nn.Linear(hidden_size, action_space))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        if self.use_actor_linear:
            return self.actor_linear(x)
        
        return x

class BCQResnetBaseEncoder(NNBase):
    """
    BCQ Netwrok with Residual Network-style encoder -> 1M parameters
    """
    def __init__(self, observation_space, action_space=15, hidden_size=64, channels=[16, 32, 32], use_actor_linear=False):
        super(BCQResnetBaseEncoder, self).__init__(hidden_size)
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size

        self.conv1 = self._make_layer(observation_space.shape[-1], channels[0])
        self.conv2 = self._make_layer(channels[0], channels[1])
        self.conv3 = self._make_layer(channels[1], channels[2])

        self.q1 = nn.Linear(2048, hidden_size)
        self.q2 = nn.Linear(hidden_size, action_space)

        self.i1 = nn.Linear(2048, hidden_size)
        self.i2 = nn.Linear(hidden_size, action_space)
        
        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        c = F.relu(self.conv1(x))
        c = F.relu(self.conv2(c))
        c = F.relu(self.conv3(c))

        q = F.relu(self.q1(c.reshape(-1, 2048)))
        i = F.relu(self.i1(c.reshape(-1, 2048)))
        i = self.i2(i)
        return self.q2(q), F.log_softmax(i, dim=1), i
```

./procgen/utils/early_stopper.py
```python
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np

class EarlyStop:
    """
    Early stopper, based on the mean return over the last "wait_epochs" epochs.
    
    Parameters
    ----------
    wait_epochs : int
        Number of epochs to wait before stopping after the mean return has not improved.
    delta : float
        Minimum improvement in mean return to consider an improvement.
    strict : bool
        If True, the wait_epochs is reset when the mean return improves.
    """
    def __init__(self, wait_epochs=1, min_delta=0.1, strict=True):
        self.wait_epochs = wait_epochs
        self.delta = min_delta
        self.strict = strict
        self.best_mean_return = -np.inf
        self.best_mean_return_epoch = 0
        self.waited_epochs = 0
    
    def should_stop(self, epoch, mean_return):
        if mean_return > self.best_mean_return + self.delta:
            self.best_mean_return = mean_return
            self.best_mean_return_epoch = epoch
            if self.strict:
                self.waited_epochs = 0
            else:
                self.waited_epochs -= 1
        else:
            self.waited_epochs += 1
        
        if self.waited_epochs >= self.wait_epochs:
            return True
        
        return False
```

./procgen/utils/filewriter.py
```python
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import csv
import datetime
import json
import logging
import os
import time
from typing import Dict

import numpy as np


def gather_metadata() -> Dict:
    date_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    # Gathering git metadata.
    try:
        import git

        try:
            repo = git.Repo(search_parent_directories=True)
            git_sha = repo.commit().hexsha
            git_data = dict(
                commit=git_sha,
                branch=None if repo.head.is_detached else repo.active_branch.name,
                is_dirty=repo.is_dirty(),
                path=repo.git_dir,
            )
        except git.InvalidGitRepositoryError:
            git_data = None
    except ImportError:
        git_data = None
    # Gathering slurm metadata.
    if "SLURM_JOB_ID" in os.environ:
        slurm_env_keys = [k for k in os.environ if k.startswith("SLURM")]
        slurm_data = {}
        for k in slurm_env_keys:
            d_key = k.replace("SLURM_", "").replace("SLURMD_", "").lower()
            slurm_data[d_key] = os.environ[k]
    else:
        slurm_data = None
    return dict(
        date_start=date_start,
        date_end=None,
        successful=False,
        git=git_data,
        slurm=slurm_data,
        env=os.environ.copy(),
    )


class FileWriter:
    def __init__(
        self,
        xpid: str = None,
        xp_args: dict = None,
        rootdir: str = "~/logs",
        symlink_to_latest: bool = True,
        seeds=None,
    ):
        if not xpid:
            # Make unique id.
            xpid = "{proc}_{unixtime}".format(proc=os.getpid(), unixtime=int(time.time()))
        self.xpid = xpid
        self._tick = 0
        self._latest_update = 0

        # Metadata gathering.
        if xp_args is None:
            xp_args = {}
        self.metadata = gather_metadata()
        # We need to copy the args, otherwise when we close the file writer
        # (and rewrite the args) we might have non-serializable objects (or
        # other unwanted side-effects).
        self.metadata["args"] = copy.deepcopy(xp_args)
        self.metadata["xpid"] = self.xpid

        formatter = logging.Formatter("%(message)s")
        self._logger = logging.getLogger("logs/out")

        # To stdout handler.
        shandle = logging.StreamHandler()
        shandle.setFormatter(formatter)
        self._logger.addHandler(shandle)
        self._logger.setLevel(logging.INFO)

        rootdir = os.path.expandvars(os.path.expanduser(rootdir))
        # To file handler.
        self.basepath = os.path.join(rootdir, self.xpid)
        if not os.path.exists(self.basepath):
            self._logger.info("Creating log directory: %s", self.basepath)
            os.makedirs(self.basepath, exist_ok=True)
        else:
            self._logger.info("Found log directory: %s", self.basepath)

        if symlink_to_latest:
            # Add 'latest' as symlink unless it exists and is no symlink.
            symlink = os.path.join(rootdir, "latest")
            try:
                if os.path.islink(symlink):
                    os.remove(symlink)
                if not os.path.exists(symlink):
                    os.symlink(self.basepath, symlink)
                    self._logger.info("Symlinked log directory: %s", symlink)
            except OSError:
                # os.remove() or os.symlink() raced. Don't do anything.
                pass

        self.paths = dict(
            msg="{base}/out.log".format(base=self.basepath),
            logs="{base}/logs.csv".format(base=self.basepath),
            fields="{base}/fields.csv".format(base=self.basepath),
            meta="{base}/meta.json".format(base=self.basepath),
            final_test_eval="{base}/final_test_eval.csv".format(base=self.basepath),
        )

        self._logger.info("Saving arguments to %s", self.paths["meta"])
        if os.path.exists(self.paths["meta"]):
            self._logger.warning("Path to meta file already exists. " "Not overriding meta.")
        else:
            self._save_metadata()

        self._logger.info("Saving messages to %s", self.paths["msg"])
        if os.path.exists(self.paths["msg"]):
            self._logger.warning("Path to message file already exists. " "New data will be appended.")

        fhandle = logging.FileHandler(self.paths["msg"])
        fhandle.setFormatter(formatter)
        self._logger.addHandler(fhandle)

        self._logger.info("Saving logs data to %s", self.paths["logs"])
        self._logger.info("Saving logs' fields to %s", self.paths["fields"])
        self.fieldnames = ["_tick", "_time"]
        self.final_test_eval_fieldnames = ["final_test_ret", "final_train_ret", "final_val_ret"]
        if os.path.exists(self.paths["logs"]):
            self._logger.warning("Path to log file already exists. " "New data will be appended.")
            # Override default fieldnames.
            with open(self.paths["fields"], "r") as csvfile:
                reader = csv.reader(csvfile)
                lines = list(reader)
                if len(lines) > 0:
                    self.fieldnames = lines[-1]
            # Override default tick: use the last tick from the logs file plus 1.
            with open(self.paths["logs"], "r") as csvfile:
                # remove null bytes in csv
                reader = csv.reader(x.replace("\0", "") for x in csvfile)
                lines = list(reader)
                # Need at least two lines in order to read the last tick:
                # the first is the csv header and the second is the first line
                # of data.
                if len(lines) > 1:
                    self._tick = int(lines[-1][0]) + 1
                    self._latest_update = int(lines[-1][lines[0].index("epoch")])

        self._fieldfile = open(self.paths["fields"], "a")
        self._fieldwriter = csv.writer(self._fieldfile)
        self._logfile = open(self.paths["logs"], "a")
        self._logwriter = csv.DictWriter(self._logfile, fieldnames=self.fieldnames)
        self._finaltestfile = open(self.paths["final_test_eval"], "a")
        self._finaltestwriter = csv.DictWriter(self._finaltestfile, fieldnames=self.final_test_eval_fieldnames)

        self._finaltestwriter.writeheader()
        self._finaltestfile.flush()

    def log(self, to_log: Dict, tick: int = None, verbose: bool = False) -> None:
        if tick is not None:
            raise NotImplementedError
        else:
            to_log["_tick"] = self._tick
            self._tick += 1
        to_log["_time"] = time.time()

        old_len = len(self.fieldnames)
        for k in to_log:
            if k not in self.fieldnames:
                self.fieldnames.append(k)
        if old_len != len(self.fieldnames):
            self._fieldwriter.writerow(self.fieldnames)
            self._logger.info("Updated log fields: %s", self.fieldnames)

        if to_log["_tick"] == 0:
            self._logfile.write("# %s\n" % ",".join(self.fieldnames))

        if verbose:
            self._logger.info(
                "LOG | %s",
                ", ".join(["{}: {}".format(k, to_log[k]) for k in sorted(to_log)]),
            )

        self._logwriter.writerow(to_log)
        self._logfile.flush()

    def log_final_test_eval(self, to_log):
        self._finaltestwriter.writerow(to_log)
        self._finaltestfile.flush()

    def close(self, successful: bool = True) -> None:
        self.metadata["date_end"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        self.metadata["successful"] = successful
        self._save_metadata()

        for f in [self._logfile, self._fieldfile]:
            f.close()

    def _save_metadata(self) -> None:
        with open(self.paths["meta"], "w") as jsonfile:
            json.dump(self.metadata, jsonfile, indent=4, sort_keys=True)

    def latest_tick(self):
        return self._tick

    def latest_update_count(self):
        return self._latest_update
```

./procgen/utils/gpt_arch.py
```python
"""
This file contains implementation of minGPT for implementing Decision Transformer.

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

----------------------------------------------------------------------------------------------------------------------------

GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                     .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        if config.inp_channels == 3:
            self.inp_state_shape = [-1, 3, 64, 64] # Procgen
        else:
            self.inp_state_shape = [-1, 4, 84, 84] # Atari

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)


        print("number of parameters in minGPT: %e", sum(p.numel() for p in self.parameters()))

        final_neurons = 3136 if config.inp_channels==4 else 1024 # Procgen
        self.state_encoder = nn.Sequential(nn.Conv2d(self.config.inp_channels, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                 nn.Flatten(), nn.Linear(final_neurons, config.n_embd), nn.Tanh())

        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, targets=None, rtgs=None, timesteps=None, padding_mask=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, 1, 1)
        # padding_mask: (batch, block_size, 1)
        state_embeddings = self.state_encoder(states.reshape(self.inp_state_shape).type(torch.float32).contiguous()) # (batch * block_size, n_embd)
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd) # (batch, block_size, n_embd)
        # multiply state embeddings by padding_mask to zero out padding states embeddings
        if padding_mask is not None:
            state_embeddings = state_embeddings * padding_mask.unsqueeze(-1).type(torch.float32)
        
        if actions is not None and self.model_type == 'reward_conditioned': 
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
            # multiply rtg embeddings by padding_mask to zero out padding rtgs embeddings
            if padding_mask is not None:
                rtg_embeddings = rtg_embeddings * padding_mask.unsqueeze(-1).type(torch.float32)
                action_embeddings = action_embeddings * padding_mask.unsqueeze(-1).type(torch.float32)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*3 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::3,:] = rtg_embeddings
            token_embeddings[:,1::3,:] = state_embeddings
            token_embeddings[:,2::3,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'reward_conditioned': # only happens at very first timestep of evaluation
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            if padding_mask is not None:
                rtg_embeddings = rtg_embeddings * padding_mask.unsqueeze(-1).type(torch.float32)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2, self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = rtg_embeddings # really just [:,0,:]
            token_embeddings[:,1::2,:] = state_embeddings # really just [:,1,:]
        elif actions is not None and self.model_type == 'naive':
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1)) # (batch, block_size, n_embd)
            if padding_mask is not None:
                action_embeddings = action_embeddings * padding_mask.unsqueeze(-1).type(torch.float32)

            token_embeddings = torch.zeros((states.shape[0], states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:,::2,:] = state_embeddings
            token_embeddings[:,1::2,:] = action_embeddings[:,-states.shape[1] + int(targets is None):,:]
        elif actions is None and self.model_type == 'naive': # only happens at very first timestep of evaluation
            token_embeddings = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd

        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
        final_input = token_embeddings + position_embeddings # (batch, block_size, n_embd)
        if padding_mask is not None:
            # multiply by padding_mask (batch_size, context_len, n_embed) where block_size = 2*context_len to zero out padding states embeddings
            repeat_padding_shape = final_input.shape[1] // padding_mask.shape[1] 
            input_padding_mask = padding_mask.repeat(1, repeat_padding_shape)
            final_input = final_input * input_padding_mask.unsqueeze(-1).type(torch.float32)
        x = self.drop(final_input)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if actions is not None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1::3, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'reward_conditioned':
            logits = logits[:, 1:, :]
        elif actions is not None and self.model_type == 'naive':
            logits = logits[:, ::2, :] # only keep predictions from state_embeddings
        elif actions is None and self.model_type == 'naive':
            logits = logits # for completeness
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=self.config.vocab_size-1)

        return logits, loss


```

./procgen/utils/plot.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import collections
import csv
import fnmatch
import os
import re
from functools import reduce
from itertools import chain, repeat
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
Usage:
    Plot grid of games:
    >>> python -m utils.plot \
            --grid  \
            -r xpid_prefix1 xpid_prefix2 \
            -l method_name1 method_name2  \
            -xi 5000000 -xts M --save_width 190 --save_height 225 \
            --linewidth=0.5 \
            -a 0.1 --fontsize 6 -yl 'Mean test episode return' \
            --savename full_procgen_results


    Comparing mean curves:
    >>> python -m utils.plot \
            --avg_procgen \
            -r \
            xpid_prefix1 \
            xpid_prefix2 \
            -l \
            method_name1 \
            method_name2 \
            -a 0.1 -xi 5000000 -xts M --save_width 200 --save_height 200 --savename 'savename'
"""

"""
Example usage:

    >>> python -m utils.plot \
            --grid  \
            -r "./data_path/offlinerl/bc" \
            --prefix easy-200-bc-p1.0-lr0.0001-bs512 \
            -l l200 \
            --x_axis=epoch \
            --y_axis=test_rets_mean \
            -xi 5000000 -xts M --save_width 190 --save_height 225 \
            --linewidth=0.5 \
            -a 0.1 --fontsize 6 -yl 'Mean test episode return' \
            --save_path . \
            --savename 'bc_test_returns_grid'

    >>> python -m utils.plot \
            --avg_procgen \
            -r "./data_path/offlinerl/bc" \
            --prefix easy-200-bc-p1.0-lr0.0001-bs512 \
            -l l200 \
            --x_axis=epoch \
            --y_axis=test_rets_mean \
            -yl 'Mean normalized test episode return' \
            -a 0.1 -xi 5000000 -xts M --save_width 200 --save_height 200 \
            --save_path . \
            --savename 'bc_test_returns_mean'
"""


class OuterZipStopIteration(Exception):
    pass


def outer_zip(*args):
    """
    https://stackoverflow.com/questions/13085861/outerzip-zip-longest-function-with-multiple-fill-values
    """
    count = len(args) - 1

    def sentinel(default):
        nonlocal count
        if not count:
            raise OuterZipStopIteration
        count -= 1
        yield default

    iters = [chain(p, sentinel(default), repeat(default)) for p, default in args]
    try:
        while iters:
            yield tuple(map(next, iters))
    except OuterZipStopIteration:
        pass


def islast(itr):
    old = next(itr)
    for new in itr:
        yield False, old
        old = new
    yield True, old


def file_index_key(f):
    pattern = r"\d+$"
    key_match = re.findall(pattern, Path(f).stem)
    if len(key_match):
        return int(key_match[0])
    return f


def reformat_large_tick_values(tick_val, pos=None):
    """
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).

    From: https://dfrieds.com/data-visualizations/how-format-large-tick-values.html
    """
    if tick_val >= 1_000_000_000:
        val = round(tick_val / 1_000_000_000, 1)
        new_tick_format = "{:}B".format(val)
    elif tick_val >= 1_000_000:
        val = round(tick_val / 1_000_000, 1)
        new_tick_format = "{:}M".format(val)
    elif tick_val >= 1000:
        val = round(tick_val / 1000, 1)
        new_tick_format = "{:}K".format(val)
    # elif tick_val < 1000 and tick_val >= 0.1:
    #    new_tick_format = round(tick_val, 1)
    elif tick_val >= 10:
        new_tick_format = round(tick_val, 1)
    elif tick_val >= 1:
        new_tick_format = round(tick_val, 2)
    elif tick_val >= 1e-4:
        # new_tick_format = '{:}m'.format(val)
        new_tick_format = round(tick_val, 3)
    elif tick_val >= 1e-8:
        # val = round(tick_val*10000000, 1)
        # new_tick_format = '{:}μ'.format(val)
        new_tick_format = round(tick_val, 8)
    else:
        new_tick_format = tick_val

    new_tick_format = str(new_tick_format)
    new_tick_format = new_tick_format if "e" in new_tick_format else new_tick_format[:6]
    index_of_decimal = new_tick_format.find(".")

    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal + 1]
        if value_after_decimal == "0" and (tick_val >= 10 or tick_val <= -10 or tick_val == 0.0):
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal + 2 :]

    # FIXME: manual hack
    if new_tick_format == "-0.019":
        new_tick_format = "-0.02"
    elif new_tick_format == "-0.039":
        new_tick_format = "-0.04"

    return new_tick_format


def gather_results_for_prefix(args, results_path, prefix, env_name: str, point_interval):
    pattern = f"*{prefix}*"

    xpids = fnmatch.filter(os.listdir(os.path.join(results_path, env_name)), pattern)
    xpids.sort(key=file_index_key)

    assert len(xpids) > 0, f"Results for {pattern} not found."

    pd_series = []

    nfiles = 0
    for i, f in enumerate(xpids):
        print(f"xpid: {f}")
        if int(f[-1]) > args.max_index:
            print("skipping xpid... ", f)
            continue
        f_in = open(os.path.join(results_path, env_name, f, args.log_filename), "rt")
        reader = csv.reader((line.replace("\0", " ") for line in f_in))
        headers = next(reader, None)
        # print(f)
        if len(headers) < 2:
            raise ValueError("result is malformed")
        headers[0] = headers[0].replace("#", "").strip()  # remove comment hash and space

        xs = []
        ys = []
        last_x = -1

        double_x_axis = False
        if "-kl" in f:
            double_x_axis = True

        # debug = False
        # if f == 'lr-ucb-hard-fruitbot-random-s500_3':
        # 	debug = True
        # 	import pdb; pdb.set_trace()

        for row_index, (is_last, row) in enumerate(islast(reader)):
            # if debug:
            # print(row_index, is_last)

            if len(row) != len(headers):
                continue

            # print(row_index)
            if args.max_lines and row_index > args.max_lines:
                break
            if row_index % point_interval == 0 or is_last:
                row_dict = dict(zip(headers, row))
                x = int(row_dict[args.x_axis])
                if args.x_axis == "step" and double_x_axis:
                    x *= 2

                if x < last_x:
                    # print(f, x, row_index)
                    continue
                last_x = x

                if args.max_x is not None and x > args.max_x:
                    print("broke here")
                    break
                if args.gap:
                    y = float(row_dict["train_rets_mean"]) - float(row_dict["test_rets_mean"])
                else:
                    try:
                        y_value = ast.literal_eval(row_dict[args.y_axis])
                        y = float(y_value[0] if isinstance(y_value, collections.abc.Container) else y_value)
                    except Exception:
                        print("setting y=None")
                        y = None

                xs.append(x)
                ys.append(y)

        pd_series.append(pd.Series(ys, index=xs).sort_index(axis=0))
        nfiles += 1

    return nfiles, pd_series


def plot_results_for_prefix(args, ax, results_path, prefix: str, label, env_name: str = None, tag=""):
    if not env_name:
        env_name = prefix.split("-")[0]

    assert env_name in PROCGEN.keys(), f"{env_name} is not a valid Procgen game!"

    nfiles, pd_series = gather_results_for_prefix(args, results_path, prefix, env_name, args.point_interval)

    for i, series in enumerate(pd_series):
        pd_series[i] = series.loc[~series.index.duplicated(keep="first")]

    try:
        df = pd.concat(pd_series, axis=1).interpolate(method="linear") * args.scale
    except Exception:
        df = pd.concat(pd_series, axis=1) * args.scale

    # TODO: HACK To prevent unwanted lines
    df.drop(df.index[-1], inplace=True)

    ewm = df.ewm(alpha=args.alpha, ignore_na=True).mean()

    all_x = np.array([i for i in df.index])
    max_x = max(all_x)
    plt_x = all_x
    plt_y_avg = np.array([y for y in ewm.mean(axis=1)])
    plt_y_std = np.array([std for std in ewm.std(axis=1)])
    # for plt_y_row in range(1, len(plt_x)):
    # 	if plt_x[plt_y_row] <= plt_x[plt_y_row-1]:
    # 		print("Error is at row",plt_y_row)
    # 		print(plt_x[plt_y_row], plt_x[plt_y_row-1])
    # print(plt_y_avg.shape)

    # max_y = max(plt_y_avg + plt_y_std)
    # y_ticks = [0, np.floor(max_y/2.), max_y]
    # ax.set_yticks(y_ticks)

    # import pdb; pdb.set_trace()

    ax.plot(plt_x, plt_y_avg, linewidth=args.linewidth, label=label)
    ax.fill_between(plt_x, plt_y_avg - plt_y_std, plt_y_avg + plt_y_std, alpha=0.1)

    if args.grid:
        ax.set_title(env_name, fontsize=args.fontsize)
    else:
        ax.title(env_name, fontsize=args.fontsize)

    info = {"max_x": max_x, "all_x": all_x, "avg_y": plt_y_avg, "std_y": plt_y_std, "df": ewm, "tag": tag}
    return info


def format_subplot(subplt):
    # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=3, prop={'size': args.fontsize})
    # tick_fontsize = 4
    tick_fontsize = 6
    subplt.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    subplt.xaxis.get_offset_text().set_fontsize(tick_fontsize)
    subplt.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(reformat_large_tick_values))
    subplt.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(mpl.ticker.FormatStrFormatter("%d")))
    subplt.tick_params(axis="y", which="major", pad=-1)
    subplt.tick_params(axis="x", which="major", pad=0)
    subplt.grid(linewidth=0.5)


def format_plot(args, fig, plt):
    ax = plt.gca()

    if args.legend_inside:
        fig.legend(loc="lower right", prop={"size": args.fontsize})
    else:
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, prop={"size": args.fontsize})
        # fig.legend(loc='upper center', bbox_to_anchor=(0.405, 0.915), ncol=1, prop={'size': args.fontsize})
        # fig.legend(loc='upper center', bbox_to_anchor=(0.65, 0.41), ncol=1, prop={'size': args.fontsize})
        # fig.legend(loc='upper center', bbox_to_anchor=(0.29, 0.95), ncol=1, prop={'size': 8})
        # fig.legend(loc='upper right', bbox_to_anchor=(1.26, 0.85), ncol=1, prop={'size': args.fontsize})
        # ax.set_title('ninja', fontsize=8)
        if args.title:
            ax.set_title(args.title, fontsize=8)

        # pass
    # ax.set_ylim([0,1.0])

    ax.set_xlabel(args.x_label, fontsize=args.fontsize)
    ax.set_ylabel(args.y_label, fontsize=args.fontsize)

    tick_fontsize = args.fontsize
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.xaxis.get_offset_text().set_fontsize(tick_fontsize)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(reformat_large_tick_values))

    if args.max_y is not None:
        ax.set_ylim(top=args.max_y)

    if args.min_y is not None:
        ax.set_ylim(bottom=args.min_y)


PROCGEN = {
    "bigfish": {"easy": (1, 40), "hard": (0, 40)},
    "bossfight": {"easy": (0.5, 13), "hard": (0.5, 13)},
    "caveflyer": {"easy": (3.5, 12), "hard": (2, 13.4)},
    "chaser": {"easy": (0.5, 13), "hard": (0.5, 14.2)},
    "climber": {"easy": (2, 12.6), "hard": (1, 12.6)},
    "coinrun": {"easy": (5, 10), "hard": (5, 10)},
    "dodgeball": {"easy": (1.5, 19), "hard": (1.5, 19)},
    "fruitbot": {"easy": (-1.5, 32.4), "hard": (-0.5, 27.2)},
    "heist": {"easy": (3.5, 10), "hard": (2, 10)},
    "jumper": {"easy": (1, 10), "hard": (1, 10)},
    "leaper": {"easy": (1.5, 10), "hard": (1.5, 10)},
    "maze": {"easy": (5, 10), "hard": (4, 10)},
    "miner": {"easy": (1.5, 13), "hard": (1.5, 20)},
    "ninja": {"easy": (3.5, 10), "hard": (2, 10)},
    "plunder": {"easy": (4.5, 30), "hard": (3, 30)},
    "starpilot": {"easy": (2.5, 64), "hard": (1.5, 35)},
}


if __name__ == "__main__":
    """
    Arguments:
            --prefix: filename prefix of result files. Results from files with shared filename prefix will be averaged.
            --results_path: path to directory with result files
            --label: labels for each curve
            --max_index: highest index i to consider in computing curves per prefix, where filenames are of form "^(prefix).*_(i)$"
            --alpha: Polyak averaging smoothing coefficient
            --x_axis: csv column name for x-axis data, defaults to "epoch"
            --y_axis: csv column name for y-axis data, defaults to "loss"
            --threshold: show a horizontal line at this y value
            --threshold_label: label for threshold
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--base_path", type=str, default="~/logs/ppo", help="base path to results directory per prefix"
    )
    parser.add_argument("-r", "--results_path", type=str, nargs="+", default=[""], help="path to results directory")
    parser.add_argument(
        "--prefix", type=str, nargs="+", default=[""], help="Plot each xpid group matching this prefix per game"
    )
    parser.add_argument(
        "--log_filename", type=str, default="logs.csv", help="Name of log output file in each result directory"
    )
    parser.add_argument("-lines", "--max_lines", type=int, default=None, help="only plot every this many points")
    parser.add_argument("--grid", action="store_true", help="Plot all prefix tuples per game in a grid")
    # parser.add_argument('--xpid_prefix', type=str, default='lr-ppo', help='Prefix of xpid folders if plotting curves aggregated by subfolders')
    parser.add_argument(
        "--xpid_prefix",
        type=str,
        nargs="+",
        default=[],
        help="Prefix of xpid folders if plotting curves aggregated by subfolders",
    )

    parser.add_argument("-s", "--scale", type=float, default=1.0, help="scale all values by this constant")
    parser.add_argument("-l", "--label", type=str, nargs="+", default=[None], help="labels")
    parser.add_argument("-m", "--max_index", type=int, default=10, help="max index of prefix match to use")

    parser.add_argument("-a", "--alpha", type=float, default=1.0, help="alpha for emwa")
    parser.add_argument("-x", "--x_axis", type=str, default="epoch", help="csv column name of x-axis data")
    parser.add_argument(
        "-y", "--y_axis", type=str, default="test:mean_episode_return", help="csv column name of y-axis data"
    )
    parser.add_argument("-yr", "--y_range", type=float, default=[], help="y range")
    parser.add_argument("-xl", "--x_label", type=str, default="Steps", help="x-axis label")
    parser.add_argument("-yl", "--y_label", type=str, default="Mean test episode return", help="y-axis label")
    parser.add_argument("-xi", "--x_increment", type=int, default=1, help="x-axis increment")
    parser.add_argument("-xts", "--x_tick_suffix", type=str, default="M", help="x-axis tick suffix")
    parser.add_argument("-pi", "--point_interval", type=int, default=1, help="only plot every this many points")
    parser.add_argument("--max_x", type=float, default=None, help="max x-value")
    parser.add_argument("--max_y", type=float, default=None, help="max y-value")
    parser.add_argument("--min_y", type=float, default=None, help="min y-value")
    parser.add_argument("--x_values_as_axis", action="store_true", help="Show exactly x-values in data along x-axis")
    parser.add_argument(
        "--ignore_x_values_in_axis", type=float, nargs="+", default=[], help="Ignore these x-values in axis"
    )
    parser.add_argument("--linewidth", type=float, default=1.0, help="line width")
    parser.add_argument("--linestyle", type=str, default="-", help="line style")

    parser.add_argument("--threshold", type=float, default=None, help="show a horizontal line at this y value")
    parser.add_argument("--threshold_label", type=str, default="", help="label for threshold")

    parser.add_argument("--save_path", type=str, default="figures/", help="Path to save image")
    parser.add_argument("--savename", type=str, default=None, help="Name of output image")
    parser.add_argument("--dpi", type=int, default=72, help="dpi of saved image")
    parser.add_argument("--save_width", type=int, default=800, help="pixel width of saved image")
    parser.add_argument("--save_height", type=int, default=480, help="pixel height of saved image")
    parser.add_argument("--fontsize", type=int, default=6, help="pixel height of saved image")
    parser.add_argument("--legend_inside", action="store_true", help="show legend inside plot")
    parser.add_argument("--title", type=str, help="title for single plot")

    parser.add_argument("--gap", action="store_true", default=False, help="Whether to plot the generalization gap")
    parser.add_argument("--avg_procgen", action="store_true", help="Average all return-normalized curves")

    parser.add_argument("--procgen_mode", type=str, default="easy", choices=["easy", "hard"], help="Procgen env mode")

    args = parser.parse_args()

    sns.set_style("whitegrid", {"grid.color": "#EFEFEF"})

    # Create an array with the colors you want to use
    # TODO: check num_colors
    num_colors = max(len(args.prefix), len(args.results_path))

    num_colors = 7
    # num_colors = 5
    # num_colors = 3
    colors = sns.husl_palette(num_colors, h=0.1)

    # tmp = colors[1]
    # colors[1] = colors[-2]
    # TODO: Why?
    tmp = colors[2]
    colors[2] = colors[-1]

    # colors[1] = colors[-1]
    # colors[2] = colors[-2]

    # colors[-1] = tmp
    # colors[1] = colors[-1]
    # colors[0] = colors[-2]

    # colors = [
    # 	(0.8859561388376407, 0.5226505841897354, 0.195714831410001), # Orange
    # 	(1., 0.19215686, 0.19215686), # TSCL red
    # 	(1, 0.7019607843137254, 0.011764705882352941), # mixreg yellow
    # 	# (0.49862995317502606, 0.6639281765667906, 0.19302982239856423), # Green
    # 		(0.20964485513246672, 0.6785281560863642, 0.6309437466865638), # Teal
    # 		(0.9615698478167679, 0.3916890619185551, 0.8268671491444017), # Pink
    # 	(0.3711152842731098, 0.6174124752499043, 0.9586047646790773), # Blue
    # ]

    # colors = [
    # 	(0.8859561388376407, 0.5226505841897354, 0.195714831410001), # Orange
    # 	# (0.49862995317502606, 0.6639281765667906, 0.19302982239856423), # Green
    # 	# (1, 0.7019607843137254, 0.011764705882352941), # mixreg yellow
    # 	# (1., 0.19215686, 0.19215686), # TSCL red
    # 	# (0.49862995317502606, 0.6639281765667906, 0.19302982239856423), # Green
    # 	(0.9615698478167679, 0.3916890619185551, 0.8268671491444017), # Pink
    # 		(0.20964485513246672, 0.6785281560863642, 0.6309437466865638), # Teal
    # 	# sub, # Blue
    # ]

    # colors =[
    # 	(0.5019607843, 0.5019607843, 0.5019607843), # gray DR
    # 	(0.8859561388376407, 0.5226505841897354, 0.195714831410001), # Orange "ROBIN":
    # 	(0.49862995317502606, 0.6639281765667906, 0.19302982239856423), # Green "OG":
    # 	(1, 0.7019607843137254, 0.011764705882352941), # mixreg yellow "Constant":
    # 	(0.9615698478167679, 0.3916890619185551, 0.8268671491444017), # Pink "Big":
    #  (0.3711152842731098, 0.6174124752499043, 0.9586047646790773), # Blue "DPD":
    # ]

    colors = [
        (0.407, 0.505, 0.850),
        (0.850, 0.537, 0.450),
        (0.364, 0.729, 0.850),
        (0.850, 0.698, 0.282),
        (0.321, 0.850, 0.694),
        (0.850, 0.705, 0.717),
        (0.4, 0.313, 0.031),
        (0.858, 0, 0.074),
        (0.098, 1, 0.309),
        (0.050, 0.176, 1),
        (0.580, 0.580, 0.580),
        (0.9615698478167679, 0.3916890619185551, 0.8268671491444017),
    ]

    # (0.20964485513246672, 0.6785281560863642, 0.6309437466865638), # Teal "Reinit":

    # Set your custom color palette
    sns.set_palette(sns.color_palette(colors))

    dpi = args.dpi
    if args.grid:
        num_plots = len(PROCGEN.keys())
        subplot_width = 4
        subplot_height = int(np.ceil(num_plots / subplot_width))
        fig, ax = plt.subplots(subplot_height, subplot_width, sharex="col", sharey=False)
        ax = ax.flatten()
        fig.set_figwidth(args.save_width / dpi)
        fig.set_figheight(args.save_height / dpi)
        fig.set_dpi(dpi)
        # fig.tight_layout()
        plt.subplots_adjust(left=0.025, bottom=0.10, right=0.97, top=0.80, wspace=0.05, hspace=0.3)
    else:
        ax = plt
        fig = plt.figure(figsize=(args.save_width / dpi, args.save_height / dpi), dpi=dpi)

    plt_index = 0
    max_x = 0

    print(f"========= Final {args.y_axis} ========")
    results_path = args.results_path
    if args.base_path:
        base_path = os.path.expandvars(os.path.expanduser(args.base_path))
        results_path = [os.path.join(base_path, p) for p in results_path]

    results_metas = list(outer_zip((results_path, results_path[-1]), (args.prefix, args.prefix[-1]), (args.label, "")))

    infos_dict = {f"{p}_{str(i)}": [] for i, p in enumerate(args.results_path)}
    for i, meta in enumerate(results_metas):
        rp, p, label = meta
        print(f"Results Path: {rp}, Prefix: {p}, Label: {label}")

        if args.grid:
            for j, env in enumerate(PROCGEN.keys()):
                if j > 0:
                    label = None
                info = plot_results_for_prefix(args, ax[j], rp, p, label, env_name=env, tag=env)
                # TODO: remove results_path[i]
                infos_dict[f"{results_path[i]}_{i}"].append(info)
                max_x = max(info["max_x"], max_x)

        elif args.avg_procgen:
            all_series = []
            for j, env in enumerate(PROCGEN.keys()):
                print(j, env)
                if j > 0:
                    label = None

                _, pd_series = gather_results_for_prefix(args, rp, p, env_name=env, point_interval=args.point_interval)

                if not args.gap:
                    R_min, R_max = PROCGEN[env][args.procgen_mode]
                    # This is the normalization part
                    pd_series = [p.add(-R_min).divide(R_max - R_min) for p in pd_series]

                all_series.append(pd_series)

            all_series_pd = []
            min_length = float("inf")
            all_series_updated = []
            for series in all_series:
                updated_series = [s[~s.index.duplicated(keep="first")] for s in series]
                all_series_updated.append([s[~s.index.duplicated(keep="first")] for s in series])
                min_length = min(np.min([len(s) for s in updated_series]), min_length)
                print(f"Length: {min(np.min([len(s) for s in updated_series]), min_length)}")

            min_length = int(min_length)
            all_series = all_series_updated

            for series in all_series:
                trunc_series = [s[:min_length] for s in series]
                all_series_pd.append(pd.concat(trunc_series, axis=1).interpolate(method="linear") * args.scale)

            df = reduce(lambda x, y: x.add(y, fill_value=0), all_series_pd) / len(PROCGEN.keys())
            # try:
            # 	import pdb; pdb.set_trace()
            # 	df = pd.concat(avg_series, axis=1).interpolate(method='linear')*args.scale
            # except:
            # 	df = pd.concat(avg_series, axis=1)*args.scale
            ewm = df.ewm(alpha=args.alpha, ignore_na=True).mean()

            all_x = np.array([i for i in df.index])
            max_x = max(all_x)
            plt_x = all_x
            plt_y_avg = np.array([y for y in ewm.mean(axis=1)])
            plt_y_std = np.array([std for std in ewm.std(axis=1, ddof=1)])

            # plot
            ax.plot(plt_x, plt_y_avg, linewidth=args.linewidth, label=meta[-1], linestyle=args.linestyle)
            ax.fill_between(plt_x, plt_y_avg - plt_y_std, plt_y_avg + plt_y_std, alpha=0.1)

            info = {
                "max_x": max_x,
                "all_x": all_x,
                "avg_y": plt_y_avg,
                "std_y": plt_y_std,
                "df": ewm,
                "tag": results_metas[i][-1],
            }
            infos_dict[f"{results_path[i]}_{i}"].append(info)
        else:
            info = plot_results_for_prefix(args, plt, rp, p, label)
            max_x = max(info["max_x"], max_x)

            # print(f"{p}: {round(info['avg_y'][-1], 2)} +/- {round(info['std_y'][-1], 2)}")

    all_x = info["all_x"]

    all_ax = ax if args.grid else [plt]
    for subax in all_ax:
        if args.threshold is not None:
            threshold_x = np.linspace(0, max_x, 2)
            subax.plot(
                threshold_x,
                args.threshold * np.ones(threshold_x.shape),
                zorder=1,
                color="k",
                linestyle="dashed",
                linewidth=args.linewidth,
                alpha=0.5,
                label=args.threshold_label,
            )

    if args.grid:
        handles, labels = all_ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1), prop={"size": args.fontsize})
        fig.text(0.5, 0.01, args.x_label, ha="center", fontsize=args.fontsize)
        fig.text(0.0, 0.5, args.y_label, va="center", rotation="vertical", fontsize=args.fontsize)
        for ax in all_ax:
            format_subplot(ax)
    else:
        format_plot(args, fig, plt)

    # Render plot
    if args.savename:
        plt.savefig(os.path.join(args.save_path, f"{args.savename}.pdf"), bbox_inches="tight", pad_inches=0, dpi=dpi)
    else:
        # plt.subplot_tool()
        plt.show()
```

./procgen/utils/utils.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import itertools
import os
import random
import sys
from enum import Enum
from typing import Dict, List

import numpy as np
import torch


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, "*.monitor.csv"))
        for f in files:
            os.remove(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LogDirType(Enum):
    CHECKPOINT = "checkpoint"
    FINAL = "final"
    ROLLING = "rolling"


class LogItemType(Enum):
    CURRENT_EPOCH = "current_epoch"
    MODEL_STATE_DICT = "model_state_dict"
    OPTIMIZER_STATE_DICT = "optimizer_state_dict"


class DatasetItemType(Enum):
    OBSERVATIONS = "observations"
    ACTIONS = "actions"
    REWARDS = "rewards"
    DONES = "dones"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def permutate_params_and_merge(grid: Dict[str, List], defaults={}) -> List[Dict[str, any]]:
    all_dynamic_combinations = permutate_values_from(grid)
    return [merge_two_dicts(defaults, dynamic_params) for dynamic_params in all_dynamic_combinations]


def permutate_values_from(grid: Dict[str, List]) -> List[Dict[str, any]]:
    """
    Example:
        >>> grid = {k_0: [v_00, v_01, v_02], k_2: [v_21, v_22]}
        >>> result = permutate_param_values(grid)
        >>> result
        [{k_0: v_00, k_2: v_21}, {k_0: v_00, k_2: v_22},
         {k_0: v_01, k_2: v_21}, {k_0: v_01, k_2: v_22},
         {k_0: v_02, k_2: v_21}, {k_0: v_02, k_2: v_22},]
    """
    params, choice_lists = zip(*grid.items())
    return [dict(zip(params, choices)) for choices in itertools.product(*choice_lists)]


def merge_two_dicts(a: Dict, b: Dict) -> Dict:
    """
    Example:
        >>> a = {k_0: v_0}
        >>> b = {k_1: v_1, k_0: v_2}
        >>> c = merge_two_dicts(a, b)
        >>> c
        {k_0: v_2, k_1: v_1}
    """
    python_version = sys.version_info
    if python_version[0] >= 3 and python_version[1] >= 9:
        return a | b
    elif python_version[0] >= 3 and python_version[1] >= 5:
        return {**a, **b}
    else:
        c = a.copy()
        c.update(b)
        return c
```

