import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import yaml
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import NatureCNN, FlattenExtractor
from models.utils.impala_cnn import ImpalaCNNLarge, ImpalaCNNSmall
from models.utils.flatten_mlp import FlattenMLP
from detached_actor_critic import DetatchedActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
import numpy as np

from data.data_generator import DataGenerator
import pdb
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import VecVideoRecorder, VecNormalize
import imageio
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import torch
from matplotlib import pyplot as plt
import re
import pandas as pd
import wandb

import argparse
import pdb


from models.policy_head.custom_callbacks import CustomEvalCallback, CustomVideoRecorder, RewardValueCallback, ValuePlottingCallback, SupervisedEncoderCallback, SelfSupervisedCovEncoderCallback, SelfSupervisedCovIKEncoderCallback, SelfSupervisedMaskEncoderCallback, SelfSupervisedMaskReconstrEncoderCallback, AdvantageLoggerCallback, InitializeLogsCallback


class PolicyHead:
    def __init__(self, model_config_path, data_config_path, seed=None):
        self.model_config = self.load_config(os.path.join(os.path.dirname(__file__), '../..', model_config_path))['policy_head']
        self.data_config = self.load_config(os.path.join(os.path.dirname(__file__), '../..', data_config_path))
        
        # self.additional_params = dict(
        #     representation_learner = self.model_config['representation_learner'] =='supervised' else False,
        #     factored = True if self.model_config['representation_learner'] == 'factored' else False
        # )
        
        self.algorithm = self.model_config['algorithm']
        self.data_type = self.data_config['observation_space']
        self.policy_name = self.select_policy()

        #set the seed in order to create argparsable separate runs for each seed
        self.seed = self.model_config['seed'] if seed is None else seed

        print('POLICY NAME: ', self.policy_name)
        

        self.parallel_train_env = VecVideoRecorder(
            self.create_parallel_envs(seed = self.seed),
            f"./logs/{self.algorithm}_{self.data_config['environment_name']}_policyviz/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/", 
            record_video_trigger=lambda x: x % (self.model_config['video_log_freq'] // self.model_config['num_parallel_envs']) == 0, 
            video_length=self.model_config['video_length'], 
            name_prefix=self.policy_name
        )

        # self.parallel_train_env = self.create_parallel_envs(seed = self.seed)

        self.valid_env = self.create_parallel_envs(seed = self.seed)
        self.eval_env = self.create_parallel_envs(seed = self.seed, train=False)

        
        self.dummy_env = self.create_env(seed=self.seed)()


        self.model = self.create_models(seed=self.seed)

        #check that critical configs for test and train are equal 
        assert (self.valid_env.observation_space == self.eval_env.observation_space), \
            f"ERROR: observaiton type {self.valid_env.observation_space} and environment {self.eval_env.observation_space} need to be same for train and eval configs"

       
        assert (self.parallel_train_env.observation_space == self.eval_env.observation_space), \
            f"ERROR: observaiton type {self.parallel_train_env.observation_space} and environment {self.eval_env.observation_space} need to be same for train and eval configs"

    def linear_schedule(self, initial_value: float):
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def select_policy(self):
        if self.model_config['learning_head'] == "supervised" or self.model_config['learning_head'].startswith('ssl'):
            return DetatchedActorCriticPolicy
        elif self.data_type == "image":
            return "CnnPolicy"
        elif self.data_type == "expert":
            return "MlpPolicy"
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def create_env(self, seed = None, config='config.yaml'):
        
        def _init():
            env = Monitor(DataGenerator(config))
            env.reset(seed=seed)
            return env

        return _init

    
    def create_parallel_envs(self, seed: int=0, train=True, num_parallel=None):
        if num_parallel is None:
            num_parallel = self.model_config['num_parallel_envs']
        if train:
            vecenv =  SubprocVecEnv([self.create_env(seed, 'config.yaml') for _ in range(num_parallel)])
        else:
            vecenv = SubprocVecEnv([self.create_env(seed, 'config_test.yaml') for _ in range(num_parallel)])
        
        #add self transposition to (C, H, W) if image observation space
        if len(vecenv.observation_space.shape) > 1:
            vecenv = VecTransposeImage(vecenv)
        
        return vecenv

    def create_models(self, seed: int = 0):
        if self.algorithm == "PPO":
            ppo_params = {k: v for k, v in self.model_config['ppo'].items() if v is not None}

            #TODO: maybe make this more elegant?
            expert_obs = self.dummy_env.expert_observation_space
            num_actions = int(self.dummy_env.action_space.n)
            
            #set the appropriate output dim
            learning_head = self.model_config['learning_head']
            if learning_head == 'direct':
                features_dim = self.model_config['ppo_policy_kwargs']['backbone_dim']
            elif learning_head == 'supervised':
                features_dim = len(expert_obs.high)
            elif 'ssl' in learning_head:
                features_dim = self.model_config['num_factors'] * self.model_config['vector_size_per_factor']

            
            policy_kwargs = dict(
                net_arch = dict(pi=self.model_config['ppo_policy_kwargs']['pi_dims'], vf=self.model_config['ppo_policy_kwargs']['vf_dims']),
                features_extractor_class = ImpalaCNNSmall if len(self.parallel_train_env.observation_space.shape) > 1 else FlattenMLP,
                features_extractor_kwargs = dict(features_dim = features_dim, backbone_dim=self.model_config['ppo_policy_kwargs']['backbone_dim'], 
                vector_size_per_factor = self.model_config['vector_size_per_factor'], num_factors = self.model_config['num_factors'], 
                expert_obs = expert_obs, learning_head = learning_head, num_actions=num_actions),
                shared_feature_extractor = True
            )
            
            #NOTE: include lr schedule if needed
            #lr_schedule = self.linear_schedule(self.model_config['learning_rate'])  
            model = PPO(
                policy=self.policy_name,
                env=self.parallel_train_env,
                seed=seed,
                tensorboard_log=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{seed}/",
                policy_kwargs = policy_kwargs,
                **ppo_params,
            )
        else:
            raise ValueError(f"Unsupported RL algorithm: {self.algorithm}")
        return model
    

    def train_and_evaluate_policy(self):
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

            
        # Use Built-in Eval Callback to support multiple parallel environments
        reward_validation_callback = CustomEvalCallback("validation", eval_env=self.valid_env, max_steps=self.data_config['max_steps'], n_eval_episodes=self.model_config['num_eval_eps'], eval_freq=self.model_config['reward_log_freq'], deterministic = True, log_path = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/", best_model_save_path = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/best_weight")
        # reward_eval_callback = CustomEvalCallback("eval", eval_env=self.eval_env, max_steps=self.data_config['max_steps'], n_eval_episodes=self.model_config['num_eval_eps'], eval_freq=self.model_config['reward_log_freq'], deterministic = True, log_path = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_tensorboard/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/", best_model_save_path = None)
       
        # VecVideoRecorder is used instead of GifLoggingCallback
        value_callback = ValuePlottingCallback(env = self.dummy_env, save_freq = self.model_config['video_log_freq']//self.model_config['num_parallel_envs'], log_dir = f"./logs/{self.algorithm}_{self.data_config['environment_name']}_policyviz/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/", num_envs= self.model_config['num_parallel_envs'], name_prefix = f'{self.policy_name}_policy_value')
        checkpoint_callback = CheckpointCallback(save_freq=self.model_config['save_weight_freq']//self.model_config['num_parallel_envs'], save_path=f"./logs/{self.algorithm}_{self.data_config['environment_name']}_weights/{self.model_config['learning_head']}_{self.data_config['observation_space']}/seed_{self.seed}/", name_prefix=f'{self.algorithm}_seed{self.seed}_step', save_replay_buffer=True)

        #Advantage Plotting and Initialization callbacks
        adv_callback = AdvantageLoggerCallback(verbose=1)
        init_callback = InitializeLogsCallback(max_steps=self.data_config['max_steps'])

        # Create the callback list
        callbacks = CallbackList([reward_validation_callback, value_callback, checkpoint_callback, adv_callback, init_callback])
        
        #If using supervised learning or our approach, create separate SupervisedEncoderCallback for supervised learning approach
        if self.model_config['learning_head'] == 'supervised':
            supervised_encoder_callback = SupervisedEncoderCallback(custom_name = "supervised")
            callbacks.callbacks.append(supervised_encoder_callback)
        
        #If using supervised learning or our approach, create separate SupervisedEncoderCallback for supervised learning approach
        if self.model_config['learning_head'] == 'ssl-cov':
            supervised_encoder_callback = SelfSupervisedCovEncoderCallback(custom_name = "ssl_covariance")
            callbacks.callbacks.append(supervised_encoder_callback)

        #If using supervised learning or our approach, create separate SupervisedEncoderCallback for supervised learning approach
        if self.model_config['learning_head'] == 'ssl-mask':
            supervised_encoder_callback = SelfSupervisedMaskEncoderCallback(custom_name = "ssl_mask")
            callbacks.callbacks.append(supervised_encoder_callback)
        
        if self.model_config['learning_head'] == 'ssl-mask-reconst':
            supervised_encoder_callback = SelfSupervisedMaskReconstrEncoderCallback(custom_name="ssl_mask_reconst")
            callbacks.callbacks.append(supervised_encoder_callback)
        
        if self.model_config['learning_head'] == 'ssl-cov-ik':
            supervised_encoder_callback = SelfSupervisedCovIKEncoderCallback(custom_name="ssl_covarience_ik")
            callbacks.callbacks.append(supervised_encoder_callback)


        self.model.learn(total_timesteps=train_interval, tb_log_name=f'{self.algorithm}_{self.seed}', progress_bar = True, reset_num_timesteps=False, callback = callbacks)

        if self.model_config['wandb_log']:
            wandb.finish()
    
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--seed', type=int, default=0)
    args = args.parse_args()
    
    
    print(DataGenerator('config.yaml').observation_space)
    print(DataGenerator('config_test.yaml').observation_space)
  
    policy_head = PolicyHead( 
        'configs/models/config.yaml', 
        'configs/data_generator/config.yaml',
        seed=args.seed
    )
    policy_head.train_and_evaluate_policy()
    


