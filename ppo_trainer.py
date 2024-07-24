import json
import os

import sumo_rl
from pettingzoo.utils import conversions
from ray.rllib.env import ParallelPettingZooEnv
from sumo_rl import SumoEnvironment
from sumo_rl.environment.env import SumoEnvironmentPZ
import env_manager
import ray
from ray.rllib.algorithms import PPOConfig, PPO
from ray import tune, air
from ray.tune.registry import register_env
import project_logger



class PPOTrainer:
    SINGLE_AGENT_ENV = "SingleAgentEnvironment"
    MULTI_AGENT_ENV = "MultiAgentEnvironment"

    def __init__(self, config_path: str, env_manager: env_manager.EnvManager, experiment_type: str):
        self._logger = project_logger.ProjectLogger(self.__class__.__name__).setup_logger()
        self.config = None
        self.config_path = config_path
        self.env_manager = env_manager
        self.env = None
        # Load all configuration data from the specified configuration file
        with open(self.config_path, 'r') as f:
            self.config_data = json.load(f)

        self.config = None
        for config in self.config_data:
            if config.get("experiment_type") == experiment_type:
                self.config_data = config.get("config")
                break

        self.experiment_type = self.config_data.get("experiment_type")
        self.env_name = self.config_data.get("env_name")
        self.train_batch_size = self.config_data["train_batch_size"]
        self.lr = self.config_data["lr"]
        self.gamma = self.config_data["gamma"]
        self.lambda_ = self.config_data["lambda_"]
        self.use_gae = self.config_data["use_gae"]
        self.grad_clip = self.config_data["grad_clip"]
        self.use_critic = self.config_data["use_critic"]
        self.entropy_coeff = self.config_data["entropy_coeff"]
        self.vf_loss_coeff = self.config_data["vf_loss_coeff"]
        self.sgd_minibatch_size = self.config_data["sgd_minibatch_size"]
        self.num_sgd_iter = self.config_data["num_sgd_iter"]
        self.log_level = self.config_data["log_level"]
        self.num_of_episodes = self.config_data["num_of_episodes"]
        self.checkpoint_freq = self.config_data["checkpoint_freq"]
        self.storage_path = self.env_manager.storage_path
        self.num_env_runners = self.config_data["num_env_runners"]
        self.kwargs = None

    def env_creator(self, env_config):
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
        ray.init(ignore_reinit_error=True)

        register_env(self.env_name, env_creator=self.env_creator)

        self.config = (PPOConfig()
                       .environment(env=self.env_name)  # TODO: needs to be taken from the env
                       .training(train_batch_size=self.train_batch_size,
                                 sgd_minibatch_size=self.sgd_minibatch_size,
                                 num_sgd_iter=self.num_sgd_iter,
                                 gamma=self.gamma,
                                 lr=self.lr,
                                 lambda_=self.lambda_,
                                 use_gae=self.use_gae,
                                 grad_clip=self.grad_clip,
                                 entropy_coeff=self.entropy_coeff,
                                 vf_loss_coeff=self.vf_loss_coeff,
                                 use_critic=self.use_critic,


                                 )
                       .env_runners(num_env_runners=self.num_env_runners, rollout_fragment_length='auto',
                                    num_envs_per_env_runner=1)
                       .learners(num_learners=self.num_env_runners)
                       .debugging(log_level=self.log_level)
                       .framework(framework="torch")
                       .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
                       .reporting(min_sample_timesteps_per_iteration=720)
                       # .evaluation(
                       #          evaluation_interval=2,
                       #          evaluation_duration=self.num_of_episodes,
                       #          evaluation_num_env_runners=1,
                       #          evaluation_duration_unit="episodes"
                       #      )
                       # .callbacks()  # TODO: Add custom callbacks as needed
                       )

        if self.env_manager.sumo_type == "MultiAgentEnvironment":
            self.config.multi_agent(policies=self.env_manager.get_policies(),
                                    policy_mapping_fn=self.env_manager.policy_mapping_fn)

        return self.config

    def train(self):
        # TODO: log with project logger the self.experiment_type
        base_config = self.config.to_dict()
        # param_space = {
        #     # Use all the base config parameters
        #     **base_config,
        #     # Override some parameters with search spaces for tuning
        #     # "lr": tune.loguniform(1e-5, 1e-1),
        #     # "gamma": tune.uniform(0.9, 0.9999),
        #     #     "lambda_": tune.uniform(0.9, 1.0),
        #     #     "clip_param": tune.uniform(0.1, 0.3),
        #     #     "entropy_coeff": tune.loguniform(1e-5, 1e-2),
        #     #     "train_batch_size": tune.choice([4000, 8000]),
        #     #     "sgd_minibatch_size": tune.choice([128, 256]),
        #     #     "num_sgd_iter": tune.randint(1, 30),
        #     #
        # }

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

    def evaluate(self, results):  # TODO: didn't get to check this function
        best_result = results.get_best_result("episode_reward_mean", "max")
        best_checkpoint = best_result.checkpoint

        config = best_result.config
        algo = PPO(config=config)

        algo.restore(best_checkpoint.path)

        # Create an instance of the environment
        # TODO: setup again the env or think of a way to get the env from the training
        env = self.env_manager.env.reset()
        for episode in range(10):  # Run 10 evaluation episodes
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0

            while not (done or truncated):
                if self.env_manager.sumo_type == self.SINGLE_AGENT_ENV:
                    action = algo.compute_single_action(obs)
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                elif self.env_manager.sumo_type == self.MULTI_AGENT_ENV:
                    actions = {}
                    for agent_id, agent_obs in obs.items():
                        policy_id = f"policy_{agent_id}"
                        actions[agent_id] = algo.compute_single_action(agent_obs, policy_id=policy_id)
                    obs, rewards, dones, infos = env.step(actions)
                    episode_reward += sum(rewards.values())
                    done = dones["__all__"]
                else:
                    raise ValueError("Invalid environment type")

            print(f"Evaluation Episode {episode} reward: {episode_reward}")
