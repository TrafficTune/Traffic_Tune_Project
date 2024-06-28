from typing import Type, Union

from stable_baselines3 import DQN
from stable_baselines3 import PPO

from sumo_rl import SumoEnvironment

from config_manager import ConfigManager
from project_logger import ProjectLogger


class ModelManagement:

    def __init__(self):
        self.logger = ProjectLogger(self.__class__.__name__).setup_logger()
        self.models_dict = {
            "DQN": DQN,
            "PPO": PPO
        }

    def save_model(self, model: Type[Union[DQN, PPO]], checkpoint_file_path: str) -> None:
        try:
            model.save(checkpoint_file_path)
            self.logger.info("Model checkpoint save successfully")
        except Exception as e:
            self.logger.error(f"Failed to save model checkpoint: {e}")

    def load_model(self, model_name: str, checkpoint_file_path: str) -> Type[Union[DQN, PPO, None]]:
        model = self.models_dict.get(model_name)
        if not model:
            self.logger.error(f"Unsupported model: {model_name}")
            return None

        try:
            model.load(checkpoint_file_path)
            self.logger.info("Model loaded successfully from checkpoint")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model from checkpoint: {e}")
            return None

    def build_agent(self, model_name: str, env: Type[SumoEnvironment]) -> Type[Union[DQN, PPO, None]]:
        agent_params = ConfigManager().load_agent_config(model_name, env)
        if not agent_params:
            self.logger.error(f"Unsupported model: {model_name}")
            return None

        model = self.models_dict.get(model_name)(**agent_params)
        self.logger.info(f"{model_name} agent built successfully")
        return model
