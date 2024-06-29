import json

import config_manager
import project_logger
from typing import Optional
from sumo_rl import SumoEnvironment
import gymnasium as gym


class EnvManagement:
    def __init__(self):
        self._logger = project_logger.ProjectLogger(self.__class__.__name__).setup_logger()
        self._envs_dict = {
            "SumoEnvironment": SumoEnvironment
        }

    def save_env(self, env: gym.Env, checkpoint_file_path: str) -> None:
        pass

    def load_env(self, env_name: str, checkpoint_file_path: str) -> Optional[SumoEnvironment]:
        pass

    def build_env(self, env_name: str) -> Optional[SumoEnvironment]:
        """
            Builds a simulation environment of a specified type using configuration parameters.

            Args:
                env_name (str): The name of the environment to build, corresponding to a key in `_envs_dict`.

            Returns:
                Optional[SumoEnvironment]: The built simulation environment if successful, or None if failed.

            Notes:
                This function loads environment configuration parameters from a configuration manager based on `env_name`.
                It then initializes a simulation environment of the specified type using the loaded parameters.
                If successful, it logs an info message indicating the successful environment build.
                If loading parameters or environment initialization fails for any reason, error messages are logged,
                and None is returned.
        """
        env_params = config_manager.ConfigManager.load_env_config(env_name)
        if not env_params:
            self._logger.error(f"Unsupported env: {env_name}")
            return None

        env_class = self._envs_dict.get(env_name)
        if not env_class:
            self._logger.error(f"Unsupported env: {env_name}")
            return None

        env = env_class(**env_params)
        self._logger.info(f"{env_name} agent built successfully")

        return env


if __name__ == "__main__":
    env_management = EnvManagement()
    env_management.save_env(SumoEnvironment("", ""), "check/check.json")
