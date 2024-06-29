from typing import Union, Optional

from stable_baselines3 import DQN, PPO

from config_manager import ConfigManager
from project_logger import ProjectLogger

from gymnasium import Env


class ModelManagement:

    def __init__(self):
        self._logger = ProjectLogger(self.__class__.__name__).setup_logger()
        self._models_dict = {
            "DQN": DQN,
            "PPO": PPO
        }

    def save_model(self, model: Union[DQN, PPO], checkpoint_file_path: str) -> None:
        try:
            model.save(checkpoint_file_path)
            self._logger.info("Model checkpoint save successfully")
        except Exception as e:
            self._logger.error(f"Failed to save model checkpoint: {e}")

    def load_model(self, model_name: str, checkpoint_file_path: str) -> Optional[DQN, PPO]:
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

    def build_agent(self, model_name: str, env: Env) -> Optional[DQN, PPO]:
        agent_params = ConfigManager().load_agent_config(model_name, env)
        if not agent_params:
            self._logger.error(f"Unsupported model: {model_name}")
            return None

        model = self._models_dict.get(model_name)(**agent_params)
        self._logger.info(f"{model_name} agent built successfully")
        return model
