import json
from time import sleep

import typer
import gymnasium as gym
from pettingzoo.utils import conversions
import ray
from ray import rllib
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import sumo_rl
from sumo_rl import SumoEnvironment
from sumo_rl.environment.env import SumoEnvironmentPZ


class EnvManager:

    def __init__(self, sumo_type: str, config_path: str):
        self.config = None
        self.config_path = config_path
        self.env = None
        self.sumo_type = sumo_type

        # Load all configuration data from the specified configuration file
        with open(self.config_path, 'r') as f:
            self.config_data = json.load(f)

        self.kwargs = self.config_data.get("kwargs")

        self._policies = {}

    def initialize_env(self):
        kwargs = self.kwargs

        if self.sumo_type == "SingleAgentEnvironment":
            self.env = SumoEnvironment(**kwargs)
        elif self.sumo_type == "MultiAgentEnvironment":
            self.env = SumoEnvironmentPZ(**kwargs)
            self.set_policies()
            self.env = conversions.aec_to_parallel(self.env)
            self.env = ParallelPettingZooEnv(self.env)
            self.env = self.env.to_base_env()
        else:
            raise TypeError("Invalid environment type, must be either 'SingleAgentEnvironment' or 'MultiAgentEnvironment'")

        return self.env

    def policy_mapping_fn(self, agent_id, episode, worker, **kwargs):
        policy_key = f"policy_{agent_id}"
        if policy_key not in self._policies:
            raise ValueError(f"Policy {policy_key} not found")
        return policy_key

    def set_policies(self):
        if self.sumo_type == "SingleAgentEnvironment":
            raise TypeError("SumoEnvironment does not support multi-agent training")

        for agent in self.env.agents:
            policy_key = f"policy_{agent}"
            obs_space = self.env.observation_space(agent)
            act_space = self.env.action_space(agent)
            self._policies[policy_key] = (None, obs_space, act_space, {})

    def get_policies(self):
        return self._policies



