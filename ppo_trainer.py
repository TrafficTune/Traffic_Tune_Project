import json
import os
import sys
import env_manager
from ray.rllib.algorithms import PPOConfig
import gymnasium as gym
from ray.tune.registry import register_env


class PPOTrainer:
    def __init__(self, config_path: str, env_manager: env_manager.EnvManager):
        self.config = None
        self.config_path = config_path
        self.env_manager = env_manager
        self.env = env_manager.env

        # Load all configuration data from the specified configuration file
        with open(self.config_path, 'r') as f:
            self.config_data = json.load(f)
        self.experiment_type = self.config_data.get("experiment_type")
        self.env_name = self.config_data.get("env_name")
        self.num_rollout_workers = self.config_data.get("num_rollout_workers")
        self.rollout_fragment_length = self.config_data.get("rollout_fragment_length")
        self.train_batch_size = self.config_data["train_batch_size"]
        self.lr = self.config_data["lr"]
        self.gamma = self.config_data["gamma"]
        self.lambda_ = self.config_data["lambda_"]
        self.use_gae = self.config_data["use_gae"]
        self.clip_param = self.config_data["clip_param"]
        self.grad_clip = self.config_data["grad_clip"]
        self.use_critic = self.config_data["use_critic"]
        self.entropy_coeff = self.config_data["entropy_coeff"]
        self.vf_loss_coeff = self.config_data["vf_loss_coeff"]
        self.sgd_minibatch_size = self.config_data["sgd_minibatch_size"]
        self.num_sgd_iter = self.config_data["num_sgd_iter"]
        self.log_level = self.config_data["log_level"]
        self.framework = self.config_data["framework"]
        self.num_gpus = self.config_data["num_gpus"]
        self.timesteps_total = self.config_data["timesteps_total"]
        self.checkpoint_freq = self.config_data["checkpoint_freq"]
        self.storage_path = self.config_data["storage_path"]
        self.policies = self.config_data["policies"]
        self.policy_mapping_fn = self.config_data["policy_mapping_fn"]
        self.num_env_runners = self.config_data["num_env_runners"]

    def env_creator(self,env_config):
        return self.env

    def build_config(self):
        register_env(self.env_name, env_creator=self.env_creator)
        self.config = (PPOConfig()
                       .environment(env=self.env_name)
                       .training(train_batch_size=self.train_batch_size,
                                 sgd_minibatch_size=self.sgd_minibatch_size,
                                 num_sgd_iter=self.num_sgd_iter,
                                 gamma=self.gamma,
                                 lr=self.lr,
                                 lambda_=self.lambda_,
                                 use_gae=self.use_gae,
                                 clip_param=self.clip_param,
                                 grad_clip=self.grad_clip,
                                 entropy_coeff=self.entropy_coeff,
                                 vf_loss_coeff=self.vf_loss_coeff,
                                 use_critic=self.use_critic
                                 )
                       .debugging(log_level=self.log_level)
                       .framework(framework=self.framework)
                       .resources(num_gpus=self.num_gpus)
                       .env_runners(num_env_runners=self.num_env_runners, rollout_fragment_length='auto')
                       .learners(num_learners=self.num_env_runners)
                       .evaluation()
                       .callbacks()
                       )

        if self.env.sumo_type == "MultiAgentEnvironment":
            self.config.multi_agent(policies=self.env_manager.get_policies(),
                                    policy_mapping_fn=self.env_manager.policy_mapping_fn)
        return self.config

    def run_tuner(self):
        pass

    # train,run,tune,fit