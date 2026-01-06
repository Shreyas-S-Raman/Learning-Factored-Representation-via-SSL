import sys
import os
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage, VecVideoRecorder, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

import torch
from data.data_generator import build_data_generator
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
import pdb
import gymnasium as gym
import wandb
import argparse

from models.representation_learning.visual import VisualRepresentationLearner
from models.representation_learning.expert import ExpertRepresentationLearner
from models.representation_learning.expert_factored_graph import FactoredGraphRepresentationLearner
from models.representation_learning.factored_vf import FactoredVFLearner
from models.representation_learning.supervised import SupervisedRepresentationLearner
from models.representation_learning.barlow_twins import BarlowTwinsRepresentationLearner
from models.policy_head.custom_callbacks import CustomEvalCallback, CustomVideoRecorder, RewardValueCallback, ValuePlottingCallback, AuxiliaryLossCallback, AdvantageLoggerCallback, InitializeLogsCallback

REPRESENTATION_LEARNERS = {
    'visual': VisualRepresentationLearner,
    'expert': ExpertRepresentationLearner,
    'factored-graph': FactoredGraphRepresentationLearner,
    'factored-vf': FactoredVFLearner,
    'supervised': SupervisedRepresentationLearner,
    'barlow-twins': BarlowTwinsRepresentationLearner,
}

class PolicyHead:
    def __init__(self, env_config_filename:str, method:str, seed:int=None):
        # setup root dir for configs
        self.config_root = os.path.join(os.path.dirname(__file__), "../../configs")
        self.model_config = self.load_config('policy_learning/config')
        
        # update method and seed within model configs
        OmegaConf.set_readonly(self.model_config, False)
        self.model_config.method = method
        if seed is not None:
            self.model_config.seed = seed
        OmegaConf.set_readonly(self.model_config, True)
        self.seed = self.model_config.seed
        
        self.data_config = self.load_config(f'env/{env_config_filename}')
        self.test_data_config = self.load_config(f'env/{self.data_config.testfile}')
        self.algorithm = self.model_config['algorithm']
        self.data_type = self.data_config['observation_space']
        self.policy_name = self.select_policy()

        self.parallel_train_env = VecVideoRecorder(
            self.create_parallel_envs(seed = self.seed, merged_config=self.data_config),
            f"./logs/{self.algorithm}_{self.data_config['environment_name']}_policyviz/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/", 
            record_video_trigger=lambda x: x % (self.model_config['video_log_freq'] // self.model_config['num_parallel_envs']) == 0, 
            video_length=self.model_config['video_length'], 
            name_prefix=self.policy_name
        )
        self.valid_env = self.create_parallel_envs(merged_config=self.data_config, seed = self.seed)
        self.eval_env = self.create_parallel_envs(merged_config=self.test_data_config, seed = self.seed)
        self.dummy_env = self.create_env(merged_config=self.data_config, seed=self.seed)()
        self.model = self.create_models(seed=self.seed)

        #check that critical configs for test and train are equal 
        assert (self.valid_env.observation_space == self.eval_env.observation_space), \
            f"ERROR: observaiton type {self.valid_env.observation_space} and environment {self.eval_env.observation_space} need to be same for train and eval configs"
        assert (self.parallel_train_env.observation_space == self.eval_env.observation_space), \
            f"ERROR: observaiton type {self.parallel_train_env.observation_space} and environment {self.eval_env.observation_space} need to be same for train and eval configs"

    def load_config(self, config_name):
        with initialize_config_dir(config_dir=self.config_root, version_base=None):
            cfg = compose(config_name=config_name)
        OmegaConf.resolve(cfg)
        OmegaConf.set_readonly(cfg, True)
        return configs

    def select_policy(self):
        if self.data_type == "image":
            return "CnnPolicy"
        elif self.data_type == "expert":
            return "MlpPolicy"
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def create_env(self, merged_config:OmegaConf, seed = None):
        def _init():
            env = Monitor(
                build_data_generator(configs=merged_config)
            )
            env.reset(seed=seed)
            return env

        return _init

    
    def create_parallel_envs(self, merged_config:OmegaConf, seed: int=0, num_parallel=None):
        if num_parallel is None:
            num_parallel = self.model_config['num_parallel_envs']
        vecenv = SubprocVecEnv([self.create_env(merged_config, seed) for _ in range(num_parallel)])
        
        #add self transposition to (C, H, W) if image observation space
        if len(vecenv.observation_space.shape) > 1:
            vecenv = VecTransposeImage(vecenv)
        
        return vecenv

    def create_models(self, seed: int = 0):
        expert_obs = self.dummy_env.expert_observation_space
        num_actions = int(self.dummy_env.action_space.n)
        
        # retrieve the relevant representation learning class
        representation_learner = REPRESENTATION_LEARNERS[self.model_config['method']]
        
        # compute the total number of scheduler invocations (for lr schedulers)
        total_scheduler_steps = self.model_config.train_interval/self.model_config.train_every\
            * self.model_config.auxiliary_loss.aux_loss_updates
        self.model_config.optimizer_params.get(self.model_config['method']).total_scheduler_steps = total_scheduler_steps
        
        features_extractor_kwargs = dict(
            observation_space = self.dummy_env.observation_space,
            obs_encoder=self.model_config.obs_encoder,
            representation_vector=self.model_config.representation_vector,
            projection_architecture=self.model_config.projection_architecture,
            rssm_configs=self.model_config.rssm_configs,
            optimizer_params=self.model_config.optimizer_params.get(
                self.model_config['method'], None),
            observation_encoder_dim = self.model_config.representation_vector.observation_encoder_dim,
            expert_obs= expert_obs,
            num_actions=num_actions
        )
        policy_kwargs = dict(
            net_arch = dict(pi=self.model_config['ppo_policy_kwargs']['pi_dims'], 
            vf=self.model_config['ppo_policy_kwargs']['vf_dims']),
            features_extractor_class = representation_learner,
            features_extractor_kwargs = features_extractor_kwargs,
            shared_feature_extractor = True
        )

        if self.algorithm == "PPO":
            ppo_params = {k: v for k, v in self.model_config['ppo'].items() if v is not None}
            model = PPO(
                policy=self.policy_name,
                env=self.parallel_train_env,
                seed=seed,
                tensorboard_log=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{seed}/",
                policy_kwargs = policy_kwargs,
                **ppo_params,
            )
        elif self.algorithm == "SAC":
            sac_params = {k:v for k,v in self.model_config['sac'].items() if v is not None}
            model = SAC(
                policy=self.policy_name,
                env=self.parallel_train_env,
                seed=seed,
                tensorboard_log=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{seed}/",
                policy_kwargs = policy_kwargs,
                **sac_params,
            )
        else:
            raise ValueError(f"Unsupported RL algorithm: {self.algorithm}")
        return model
    

    def train_and_evaluate_policy(self):
        if self.model_config.get('log_wandb', False):
            wandb.init(
                project='disentangled_representations',
                entity='ssl-factored-reps', 
                name=f'{self.algorithm}_{self.data_config["environment_name"]}_{self.data_config["observation_space"]}_seed_{self.seed}',
                group=f'{self.algorithm}_{self.data_config["environment_name"]}_{self.data_config["observation_space"]}',
                sync_tensorboard=True,
                monitor_gym=True,
                config={
                    "model": self.model_config,
                    "data": self.data_config,
                    "seed": self.seed,
                    "num_parallel_envs": self.model_config['num_parallel_envs']
                }
            )
        train_interval = self.model_config['train_interval']

        if os.path.exists(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}") and len(os.listdir(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}")) > 0:
            try:
                #fetch the best weights for the model rather than latest
                best_weight = os.listdir(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/best_weight")[0].split('.')[0]
                final_path = os.path.join(f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/best_weight", best_weight)

                self.model.load(path = final_path, env = self.parallel_train_env)
            except:
                pass

            
        # use Built-in Eval Callback to support multiple parallel environments
        reward_validation_callback = CustomEvalCallback("validation", eval_env=self.valid_env, max_steps=self.data_config['max_steps'], n_eval_episodes=self.model_config['num_eval_eps'], eval_freq=self.model_config['reward_log_freq'], deterministic = True, log_path = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/", best_model_save_path = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/best_weight")
        # reward_eval_callback = CustomEvalCallback("eval", eval_env=self.eval_env, max_steps=self.data_config['max_steps'], n_eval_episodes=self.model_config['num_eval_eps'], eval_freq=self.model_config['reward_log_freq'], deterministic = True, log_path = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/", best_model_save_path = None)
       
        # vecVideoRecorder is used instead of GifLoggingCallback
        value_callback = ValuePlottingCallback(env = self.dummy_env, save_freq = self.model_config['video_log_freq']//self.model_config['num_parallel_envs'], log_dir = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_policyviz/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/", num_envs= self.model_config['num_parallel_envs'], name_prefix = f'{self.policy_name}_policy_value')
        checkpoint_callback = CheckpointCallback(save_freq=self.model_config['save_weight_freq']//self.model_config['num_parallel_envs'], save_path=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/", name_prefix=f'{self.algorithm}_seed{self.seed}_step', save_replay_buffer=True)

        # advantage Plotting and Initialization callbacks
        adv_callback = AdvantageLoggerCallback(verbose=1)
        init_callback = InitializeLogsCallback(max_steps=self.data_config['max_steps'])

        # create custom callbacks for loss function
        auxiliary_loss_callback = AuxiliaryLossCallback(
            custom_name=self.model_config['method'],
            num_envs=self.model_config['num_parallel_envs'],
            train_every = self.model_config['auxiliary_loss']['train_every'],
            batch_size = self.model_config['optimizer_params'][self.model_config['method']]['batch_size'],
            learning_rate = self.model_config['auxiliary_loss']['learning_rate'],
            aux_loss_updates = self.model_config['auxiliary_loss']['aux_loss_updates'],
            verbose = 0
        )

        # create the callback list
        callbacks = CallbackList([
            reward_validation_callback,
            value_callback,
            checkpoint_callback,
            adv_callback,
            init_callback,
            auxiliary_loss_callback,
        ])
        
        self.model.learn(total_timesteps=train_interval, tb_log_name=f'{self.algorithm}_{self.seed}', progress_bar = True, reset_num_timesteps=False, callback = callbacks)
        if self.model_config.get('log_wandb', False):
            wandb.finish()
    
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--seed', type=int, default=0)
    args.add_argument('--env_config_filename', '-f', type=str, default=None)
    args.add_argument('--method', '-m', type=str, default=None)
    args = args.parse_args()
    
    policy_head = PolicyHead( 
        env_config_filename=args.env_config_filename,
        seed=args.seed,
        method=args.method
    )
    policy_head.train_and_evaluate_policy()
    


