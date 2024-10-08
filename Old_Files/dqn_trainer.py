import json
import os

from sumo_rl import SumoEnvironment
import env_manager
import ray
from ray.rllib.algorithms import DQNConfig, DQN
from ray import tune, air
from ray.tune.registry import register_env
from Logs import project_logger


class DQNTrainer:
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
        self.grad_clip = self.config_data["grad_clip"]
        self.target_network_update_freq = self.config_data["target_network_update_freq"]
        self.store_buffer_in_checkpoints = self.config_data["store_buffer_in_checkpoints"]
        self.adam_epsilon = self.config_data["adam_epsilon"]
        self.v_max = self.config_data["v_max"]
        self.v_min = self.config_data["v_min"]
        self.num_atoms = self.config_data["num_atoms"]
        self.noisy = self.config_data["noisy"]
        self.sigma0 = self.config_data["sigma0"]
        self.dueling = self.config_data["dueling"]
        self.double_q = self.config_data["double_q"]
        self.hiddens = self.config_data["hiddens"]
        self.n_step = self.config_data["n_step"]
        self.training_intensity = self.config_data["training_intensity"]

        self.log_level = self.config_data["log_level"]
        self.num_of_episodes = self.config_data["num_of_episodes"]
        self.checkpoint_freq = self.config_data["checkpoint_freq"]
        self.num_env_runners = self.config_data["num_env_runners"]
        self.storage_path = self.env_manager.storage_path
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

        self.config = (DQNConfig()
                       .environment(env=self.env_name)  # TODO: needs to be taken from the env
                       .training(
                                train_batch_size=self.train_batch_size,
                                lr=self.lr,
                                gamma=self.gamma,
                                grad_clip=self.grad_clip,
                                target_network_update_freq=self.target_network_update_freq,
                                store_buffer_in_checkpoints=self.store_buffer_in_checkpoints,
                                adam_epsilon=self.adam_epsilon,
                                v_max=self.v_max,
                                v_min=self.v_min,
                                num_atoms=self.num_atoms,
                                noisy=self.noisy,
                                sigma0=self.sigma0,
                                dueling=self.dueling,
                                double_q=self.double_q,
                                hiddens=self.hiddens,
                                n_step=self.n_step,
                                training_intensity=self.training_intensity,
                                num_steps_sampled_before_learning_starts=1
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
                       # periodical evaluation during training
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
                stop={"training_iteration": self.num_of_episodes * self.num_env_runners},
                # each iteration in multi agent there are 1002 steps
                # TODO: Stop condition need to change
                #  in single, runs 4 episodes instead of 3
                #  in multi, currently does not stop
                #  (ran about 3 episodes instead of 2, did not stop on its own)
            )
        )

        results = tuner.fit()
        return results

    def evaluate(self, results):  # TODO: didn't get to check this function
        best_result = results.get_best_result("episode_reward_mean", "max")
        best_checkpoint = best_result.checkpoint

        config = best_result.config
        algo = DQN(config=config)

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


if __name__ == "__main__":
    num_intersection = 3  # Choose which intersection you want to train

    # Choose the experiment_type:
    # PPO_SingleAgent | PPO_MultiAgent | DQN_SingleAgent | DDQN_SingleAgent | DQN_MultiAgent | DDQN_MultiAgent
    experiment_type = "DDQN_SingleAgent"

    num_training_cycles = 1
    # Initialize the environment manager
    manager = env_manager.EnvManager(f"SingleAgentEnvironment", "../Config/env_config.json", json_id=f"intersection_{num_intersection}")
    generator = manager.env_generator(
        f"Nets/intersection_{num_intersection}/route_xml_path_intersection_{num_intersection}.txt",
        algo_name="ppo")

    # Initialize the environment manager with new route file
    rou, csv = next(generator)
    manager.initialize_env(rou, csv)

    ppo_agent = ALGOTrainer(config_path="../Config/dqn_config.json", env_manager=manager, experiment_type=experiment_type)

    print(ppo_agent.experiment_type)
    print(ppo_agent.env_name)
    print(ppo_agent.num_env_runners)