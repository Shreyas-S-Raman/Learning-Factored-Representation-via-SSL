import yaml
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
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
from models.policy_head.policy_head import PolicyHead

import argparse



from custom_callbacks import CustomEvalCallback, CustomVideoRecorder, RewardValueCallback, ValuePlottingCallback


class EvaluatePolicy:

    def __init__(self, model_config_path, data_config_path, min_seed=None, max_seed=None):

        self.policy_head = PolicyHead(model_config_path, data_config_path, seed=min_seed)

        self.min_seed = min_seed
        self.max_seed = max_seed

        #load specific pre-learned weights to the policy learning model
        self.policy_head.load(path=self.policy_head.data_config['policy_weights_checkpoint'], env=self.policy_head.parallel_train_env)

        #check the observation space of environment used to learn the policy and the current environment we are evaluating
        train_obs_shape = self.get_trained_env_obs_shape(self.policy_head.data_config['train_environment_name'])
        test_obs_shape = self.policy_head.valid_env.env[0].unwrapped.get_frame(tile_size=8).shape

        self.transform, self.transform_func = self.transform_visual_observations(train_obs_shape, test_obs_shape)
        

    def get_trained_env_obs_shape(self, trained_env:str):
        dummy_env = gym.make(trained_env, render_mode='rgb_array')
        dummy_env = FullyObsWrapper(dummy_env)
        dummy_env = ImgObsWrapper(dummy_env)
        dummy_env = TimeLimit(dummy_env, max_episode_steps=configs['max_steps'])
        # Wrap the environment to enable stochastic actions
        if configs['deterministic_action'] is False:
            dummy_env = StochasticActionWrapper(env=dummy_env, prob=configs['action_stochasticity'])
         # Reset the environment to initialize it
        dummy_env.reset()
        obs_shape = dummy_env.unwrapped.get_frame(tile_size=8).shape

        return obs_shape

    def transform_visual_observations(self, train_obs_shape, test_obs_shape):

        #check if transformation needed
        if self.policy_head.data_config['observation_space'] == 'image' and train_obs_shape != test_obs_shape:
            #TODO: implement transformation function
            return True, None
        else:
            return False, None


    def run_policy_evaluation(self):

        mean_reward, std_reward, mean_success, std_success, mean_steps, std_steps = self.policy_head.evaluate_policy(self.min_seed, self.max_seed, self.transform, self.transform_func)

        print(f"Mean Reward: {mean_reward}±{std_reward}")
        print("---------------")
        print(f"Mean Success: {mean_success}±{std_success}")
        print("---------------")
        print(f"Mean Steps: {mean_steps}±{std_steps}")



