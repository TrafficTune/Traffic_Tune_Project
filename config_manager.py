import json
from typing import Optional

import project_logger
import gymnasium as gym


class ConfigManager:
    def __init__(self):
        self._logger = project_logger.ProjectLogger(self.__class__.__name__).setup_logger()

    def load_env_config(self, env_config_file_path: str) -> Optional[dict]:
        """
                Load environment configuration from a JSON file.

                Args:
                    env_config_file_path (str): Path to the environment configuration JSON file.

                Returns:
                    Optional[dict]: Dictionary containing environment configuration parameters
                                    if loaded successfully, or None if failed.

                Notes:
                    This method attempts to load environment configuration parameters from a JSON file
                    located at the specified file path. It expects the file to be in JSON format.
                    If successful, it logs an info message indicating successful loading.
                    If the file format is not '.json' or if any exception occurs during loading,
                    it logs an error message with the encountered exception and returns None.
            """
        try:
            if env_config_file_path.endswith('.json'):
                with open(env_config_file_path, 'r') as f:
                    env_dict = json.load(f)
            else:
                raise ValueError("Unsupported config file format")

            self._logger.info("Environment configuration loaded successfully")
            return env_dict
        except Exception as e:
            self._logger.error(f"Failed to load environment config: {e}")
            return None

    def load_agent_config(self, agent_config_file_path: str, env: gym.Env) -> Optional[dict]:
        """
                Load agent configuration from a JSON file and include the environment instance.

                Args:
                    agent_config_file_path (str): Path to the agent configuration JSON file.
                    env (gym.Env): Gym environment instance associated with the agent.

                Returns:
                    Optional[dict]: Dictionary containing agent configuration parameters
                                    with the 'env' key set to the provided Gym environment instance,
                                    if loaded successfully, or None if failed.

                Notes:
                    This method attempts to load agent configuration parameters from a JSON file
                    located at the specified file path. It expects the file to be in JSON format.
                    After loading, it adds the provided Gym environment instance (`env`) to the
                    dictionary under the key 'env'. If successful, it logs an info message indicating
                    successful loading. If the file format is not '.json' or if any exception occurs
                    during loading, it logs an error message with the encountered exception and returns None.
            """
        try:
            if agent_config_file_path.endswith('.json'):
                with open(agent_config_file_path, 'r') as f:
                    agent_dict: dict = json.load(f)
            else:
                raise ValueError("Unsupported config file format")

            agent_dict["env"] = env
            self._logger.info("Agent configuration loaded successfully")

            return agent_dict
        except Exception as e:
            self._logger.error(f"Failed to load agent config: {e}")
            return None
