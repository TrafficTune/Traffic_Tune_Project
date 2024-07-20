import json
from datetime import datetime
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

    def __init__(self, sumo_type: str, config_path: str, json_index: int):
        self.config = None
        self.config_path = config_path
        self.env = None
        self.sumo_type = sumo_type

        # Load all configuration data from the specified configuration file
        with open(self.config_path, 'r') as f:
            self.config_data = json.load(f)

        self.kwargs = self.config_data[json_index].get("kwargs")

        self._policies = {}

    def initialize_env(self, route: str, csv_file: str):
        kwargs = self.kwargs
        kwargs["route_file"] = route
        kwargs["out_csv_name"] = csv_file

        return kwargs
    
    def policy_mapping_fn(self, agent_id, episode, worker, **kwargs):
        policy_key = f"policy_{agent_id}"
        if policy_key not in self._policies:
            raise ValueError(f"Policy {policy_key} not found")
        return policy_key

    def set_policies(self):
        if self.sumo_type == "SingleAgentEnvironment":
            raise TypeError("SumoEnvironment does not support multi-agent training")
        self.env = SumoEnvironmentPZ(**self.kwargs)
        for agent in self.env.agents:
            policy_key = f"policy_{agent}"
            obs_space = self.env.observation_space(agent)
            act_space = self.env.action_space(agent)
            self._policies[policy_key] = (None, obs_space, act_space, {})
        self.env.close()

    def get_policies(self):
        return self._policies

    def env_generator(self, rou_path: str):
        with open(rou_path, 'r') as rou_file:
            for rou_line in rou_file:
                unique_id = datetime.now().strftime("%m.%d-%H:%M:%S")
                csv_output_path = self.generate_csv_output_path(rou_line.strip(), unique_id)
                yield rou_line.strip(), csv_output_path

    def generate_csv_output_path(self, rou_line: str, unique_id: str):
        if rou_line.startswith('Nets/'):
            base_output = rou_line.replace('Nets/', 'Outputs/Training/', 1)
            path_parts = base_output.split('/')
            rou_id = path_parts[4].split('.')
            csv_output_path = f"{path_parts[0]}/{path_parts[1]}/{path_parts[2]}/experiments/{rou_id[0]}_{unique_id}"
            return csv_output_path
        else:
            raise ValueError("Invalid route file path")
