from project_logger import ProjectLogger
from stable_baseline3 import DQN
from stable_baseline3 import PPO

from config_manager import ConfigManager


class ModelManagment:

    def __init__(self):
        self.logger = ProjectLogger(self.__class__.__name__).setup_logger()
        self.models_dict = {
            "DQN": DQN,
            "PPO": PPO
        }

    def save_model(self, model, checkpoint_file_path):
        pass

    def load_model(self, checkpoint_file_path):
        pass

    def build_agent(self, model_name, env):
        agent_params = ConfigManager().load_agent_config(model_name, env)
        if not agent_params:
            self.logger.error(f"Unsupported model: {model_name}")
            return None

        model = self.models_dict.get(model_name)(**agent_params)
        self.logger.info(f"{model_name} agent built successfully")
        return model