import config_manager
import project_logger
import gymnasium as gym

from typing import Union, Optional
from stable_baselines3 import DQN, PPO




class ModelManagement:

    def __init__(self):
        self._logger = project_logger.ProjectLogger(self.__class__.__name__).setup_logger()
        self._models_dict = {
            "DQN": DQN,
            "PPO": PPO
        }

    def save_model(self, model: Union[DQN, PPO], checkpoint_file_path: str) -> None:
        """
            Saves the given reinforcement learning model to a specified checkpoint file.

            Args:
                model (Union[DQN, PPO]): The reinforcement learning model to be saved.
                checkpoint_file_path (str): The file path where the checkpoint will be saved.

            Returns:
                None

            Notes:
                This function saves the model using its built-in save method. If the save operation
                fails for any reason, an error message is logged.
        """
        try:
            model.save(checkpoint_file_path)
            self._logger.info("Model checkpoint save successfully")
        except Exception as e:
            self._logger.error(f"Failed to save model checkpoint: {e}")

    def load_model(self, model_name: str, checkpoint_file_path: str) -> Optional[DQN, PPO]:
        """
            Loads a reinforcement learning model from a specified checkpoint file.

            Args:
                model_name (str): The name of the model to load. Should match a key in `_models_dict`.
                checkpoint_file_path (str): The file path where the model checkpoint is saved.

            Returns:
                Optional[Union[DQN, PPO]]: The loaded reinforcement learning model if successful, or None if failed.

            Notes:
                This function attempts to load the model specified by `model_name` from the checkpoint file located
                at `checkpoint_file_path`. If successful, it logs an info message indicating the successful load.
                If loading fails for any reason, an error message is logged, and None is returned.
        """
        model_class = self._models_dict.get(model_name)
        if not model_class:
            self._logger.error(f"Unsupported model: {model_name}")
            return None

        try:
            model = model_class.load(checkpoint_file_path)
            self._logger.info("Model loaded successfully from checkpoint")
            return model
        except Exception as e:
            self._logger.error(f"Failed to load model from checkpoint: {e}")
            return None

    def build_agent(self, model_name: str, env: gym.Env) -> Optional[DQN, PPO]:
        """
            Builds a reinforcement learning agent of a specified model type using provided environment.

            Args:
                model_name (str): The name of the model to build, corresponding to a key in `_models_dict`.
                env (gym.Env): The Gym environment to be used by the agent.

            Returns:
                Optional[Union[DQN, PPO]]: The built reinforcement learning agent if successful, or None if failed.

            Notes:
                This function loads agent parameters from a configuration file based on `model_name` and `env`.
                It then initializes a reinforcement learning agent of the specified model type using the loaded
                parameters and the provided environment (`env`). If successful, it logs an info message indicating
                the successful agent build. If loading parameters or agent initialization fails for any reason,
                an error message is logged, and None is returned.
        """
        agent_params = config_manager.ConfigManager().load_agent_config(model_name, env)
        if not agent_params:
            self._logger.error(f"Unsupported model: {model_name}")
            return None

        model = self._models_dict.get(model_name)(**agent_params)
        self._logger.info(f"{model_name} agent built successfully")
        return model
