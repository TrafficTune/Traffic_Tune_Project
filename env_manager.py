import json
from datetime import datetime
import os
from pettingzoo.utils import conversions
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from sumo_rl.environment.env import SumoEnvironmentPZ


class EnvManager:
    """
    A class to manage SUMO environments for reinforcement learning.
    This class handles the initialization and configuration of SUMO environments,
    including both single-agent and multi-agent scenarios.
    """

    def __init__(self, sumo_type: str, config_path: str, intersection_id: str):
        """
        Initialize the EnvManager.
        Args:
            sumo_type (str): Type of SUMO environment ('SingleAgentEnvironment' | 'MultiAgentEnvironment').
            config_path (str): Path to the env configuration file.
            intersection_id (str): Identifier for the specific configuration in the JSON file.
        """
        self.config = None
        self.config_path = config_path
        self.env = None
        self.sumo_type = sumo_type
        self.storage_path = None

        # Load all configuration data from the specified configuration file
        with open(self.config_path, 'r') as f:
            self.config_data = json.load(f)

        self.kwargs = None
        for config in self.config_data:
            if config.get("intersection_id") == intersection_id:
                self.kwargs = config.get("kwargs")
                break

        self._policies = {}

    def initialize_env(self, route: str, csv_file: str):
        """
        Initialize the SUMO environment with specific route and output CSV file.

        Args:
            route (str): Path to the route file.
            csv_file (str): Path for the output CSV file.

        Returns:
            dict: Updated kwargs for environment initialization.
        """
        kwargs = self.kwargs
        net_path = kwargs["net_file"]
        abspath = os.path.dirname(os.path.abspath(__file__))
        kwargs["net_file"] = f"{abspath}/{net_path}"
        kwargs["route_file"] = route
        kwargs["out_csv_name"] = csv_file
        if self.sumo_type == "MultiAgentEnvironment":
            self.set_policies()
        self.storage_path = self.get_storage_path(csv_file)
        return kwargs

    def policy_mapping_fn(self, agent_id, episode, worker, **kwargs):
        """
        Map agent IDs to policy keys for multi-agent only.

        Args:
            agent_id: The ID of the agent.
            episode: The current episode.
            worker: The current worker.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The policy key for the given agent ID.

        Raises:
            ValueError: If the policy for the given agent ID is not found.
        """
        policy_key = f"policy_{agent_id}"
        if policy_key not in self._policies:
            raise ValueError(f"Policy {policy_key} not found")
        return policy_key

    def set_policies(self):
        """
        Set up policies for multi-agent environments.

        This method creates policies for each agent,
        and converts the environment to a parallel PettingZoo environment.

        Raises:
            TypeError: If attempting to set policies for a single-agent environment.

        Note:
            - This method should only be called for multi-agent environments.
            - It initializes the environment using SumoEnvironmentPZ.
            - For each agent, it creates a policy with a specific observation and action spaces.
            - The environment is converted to a parallel environment for compatibility with RLlib.
        """
        if self.sumo_type == "SingleAgentEnvironment":
            raise TypeError("SumoEnvironment does not support multi-agent training")

        # Initialize the SUMO environment
        self.env = SumoEnvironmentPZ(**self.kwargs)

        # Create policies for each agent (tls)
        for agent in self.env.agents:
            policy_key = f"policy_{agent}"
            obs_space = self.env.observation_space(agent)
            act_space = self.env.action_space(agent)
            # Store policy tuple: (policy, obs_space, act_space, config)
            self._policies[policy_key] = (None, obs_space, act_space, {})

        self.env = conversions.aec_to_parallel(self.env)
        self.env = ParallelPettingZooEnv(self.env)
        self.env = self.env.to_base_env()

    def get_policies(self):
        """
        Get the current policies.

        Returns:
            dict: The current policies.
        """
        return self._policies

    def env_generator(self, rou_path: str, algo_name: str):
        """
        Generate environments based on route files.

        Args:
            rou_path (str): Path to the route file.
            algo_name (str): Name of the algorithm being used.

        Yields:
            tuple: A tuple containing the route line and the CSV output path.
        """
        with open(rou_path, 'r') as rou_file:
            for rou_line in rou_file:
                unique_id = datetime.now().strftime("%m.%d-%H:%M:%S")
                csv_output_path = self.generate_csv_output_path(rou_line.strip(), unique_id, algo_name)
                yield rou_line.strip(), csv_output_path

    def generate_csv_output_path(self, rou_line: str, unique_id: str, algo_name: str):
        """
        Generate the output path for CSV files.

        Args:
            rou_line (str): The route line from the route file.
            unique_id (str): A unique identifier for the output CSV.
            algo_name (str): Name of the algorithm being used.

        Returns:
            str: The generated CSV output path.

        Raises:
            ValueError: If the route file path is invalid.
        """
        if rou_line.startswith('Nets/'):
            base_output = rou_line.replace('Nets/', 'Outputs/Training/', 1)
            path_parts = base_output.split('/')
            rou_id = path_parts[4].split('.')
            csv_output_path = f"{path_parts[0]}/{path_parts[1]}/{path_parts[2]}/experiments/{algo_name}_{rou_id[0]}_{unique_id}"
            return csv_output_path
        else:
            raise ValueError("Invalid route file path")

    def get_storage_path(self, csv_file: str):
        """
        Get the storage path for saved agents.

        Args:
            csv_file (str): The CSV file path.

        Returns:
            str: The storage path for saved agents.
        """
        if csv_file.startswith('Outputs/Training/'):
            path_parts = csv_file.split('/')
            abspath = os.path.dirname(os.path.abspath(__file__))
            storage_path = f"{abspath}/{path_parts[0]}/{path_parts[1]}/{path_parts[2]}/saved_agent"
            return storage_path
