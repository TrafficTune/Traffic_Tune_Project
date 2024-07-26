import json
import os
from sumo_rl import SumoEnvironment
import env_manager
import ray
from ray.rllib.algorithms import PPOConfig, PPO
from ray.rllib.algorithms import DQNConfig, DQN
from ray import tune, air
from ray.tune.registry import register_env
import project_logger


class ALGOTrainer:
    """
    A class for training reinforcement learning algorithms on SUMO environments.

    This class handles the configuration and execution of training for both
    single-agent and multi-agent environments using PPO | DQN | DDQN algorithms.
    """

    SINGLE_AGENT_ENV = "SingleAgentEnvironment"
    MULTI_AGENT_ENV = "MultiAgentEnvironment"

    def __init__(self, config_path: str, env_manager: env_manager.EnvManager, experiment_type: str):
        """
        Initialize the ALGOTrainer.

        Args:
            config_path (str): Path to the algorithm configuration file ppo | dqn.
            env_manager (env_manager.EnvManager): An instance of the EnvManager class.
            experiment_type (str): Type of experiment
            (PPO_SingleAgent | DQN_SingleAgent | DDQN_SingleAgent | PPO_MultiAgent | DQN_MultiAgent | DDQN_MultiAgent).

        Raises:
            ValueError: If the experiment_type and config_path do not match.
        """
        self._logger = project_logger.ProjectLogger(self.__class__.__name__).setup_logger()
        self.env_manager = env_manager
        self.env = None
        self.config_path = config_path

        if not self.is_valid_experiment(experiment_type=experiment_type, config_path=config_path):
            raise ValueError("ALGOTrainer: The experiment_type and the config_path do not match\n"
                             "Please make sure both match the same algorithm (DQN | PPO)")

        # Load all configuration data from the specified algorithm configuration file
        with open(self.config_path, 'r') as f:
            self.config_data = json.load(f)

        self.config = None
        for config in self.config_data:
            if config.get("experiment_type") == experiment_type:
                self.config = config
                break

        # Initialize attributes from the configuration
        self.experiment_type = self.config["experiment_type"]
        self.env_name = self.config["env_name"]
        self.log_level = self.config["log_level"]
        self.num_of_episodes = self.config["num_of_episodes"]
        self.checkpoint_freq = self.config["checkpoint_freq"]
        self.num_env_runners = self.config["num_env_runners"]
        self.training_config = self.config["config"]
        self.storage_path = self.env_manager.storage_path
        self.kwargs = None

        # Set the algorithm configuration class based on the experiment type
        self.ALGOConfig = PPOConfig if "PPO" in self.experiment_type else DQNConfig

    def env_creator(self, env_config):
        """
        Create and return the appropriate environment based on the SUMO type.

        Args:
            env_config: Configuration for the environment (not used in this method).

        Returns:
            The created environment.

        Raises:
            TypeError: If an invalid environment type is specified.
        """
        if self.env_manager.sumo_type == self.SINGLE_AGENT_ENV:
            env = SumoEnvironment(**self.env_manager.kwargs)
            self.env_manager.env = env
        elif self.env_manager.sumo_type == self.MULTI_AGENT_ENV:
            env = self.env_manager.env
        else:
            raise TypeError(
                "Invalid environment type, must be either 'SingleAgentEnvironment' or 'MultiAgentEnvironment'")
        return env

    def build_config(self):
        """
        Build and return the configuration for the training algorithm.

        This method initializes Ray, registers the environment, and sets up the
        configuration for the chosen algorithm (PPO or DQN).

        Returns:
            The configured algorithm configuration.
        """
        ray.init(ignore_reinit_error=True)

        register_env(self.env_name, env_creator=self.env_creator)

        self.config = (self.ALGOConfig()
                       .environment(env=self.env_name)
                       .training(**self.training_config)
                       .env_runners(num_env_runners=self.num_env_runners, rollout_fragment_length='auto',
                                    num_envs_per_env_runner=1)
                       .learners(num_learners=self.num_env_runners)
                       .debugging(log_level=self.log_level)
                       .framework(framework="torch")
                       .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
                       .reporting(min_sample_timesteps_per_iteration=720)
                       # .evaluation(
                       #         evaluation_interval=2,
                       #         evaluation_duration=5,
                       #         evaluation_force_reset_envs_before_iteration=True,
                       #         evaluation_num_env_runners=1,
                       #         evaluation_duration_unit="episodes",
                       #         evaluation_parallel_to_training=True
                       #     )
                       # .callbacks() TODO: Add a custom callbacks as needed
                       )

        if self.env_manager.sumo_type == self.MULTI_AGENT_ENV:
            self.config.multi_agent(policies=self.env_manager.get_policies(),
                                    policy_mapping_fn=self.env_manager.policy_mapping_fn)

        return self.config

    def train(self):
        """
        Execute the training process.

        This method sets up the tuner with the specified configuration and runs
        the training process.

        Returns:
            The results of the training process.
        """
        base_config = self.config.to_dict()

        tuner = tune.Tuner(
            self.env_name,
            param_space=base_config,
            run_config=air.RunConfig(
                storage_path=self.storage_path,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_at_end=True,
                    checkpoint_frequency=self.checkpoint_freq,
                    checkpoint_score_attribute='env_runners/episode_reward_max',
                    checkpoint_score_order="max"
                ),
                stop={'training_iteration': self.num_of_episodes * self.num_env_runners},
            )
        )

        results = tuner.fit()
        return results

    def is_valid_experiment(self, experiment_type, config_path):
        """
        Check if the experiment type matches the configuration file path.

        Args:
            experiment_type (str): The type of experiment (e.g., "PPO", "DQN").
            config_path (str): The path to the configuration file.

        Returns:
            bool: True if the experiment type matches the config path, False otherwise.
        """
        if ("PPO" in experiment_type and "ppo" in config_path) or ("DQN" in experiment_type and "dqn" in config_path):
            return True
        return False
