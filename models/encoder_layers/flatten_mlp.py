from __future__ import annotations
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import gymnasium as gym
import torch
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import pdb
import torch.nn.functional as F
from typing import Optional

class FlattenMLP(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, 
                 observation_space: gym.Space, 
                 features_dim:int=256, 
                 expert_obs: gym.Space = None, 
                 num_factors: int = None, 
                 num_actions: int = None,
                 vector_size_per_factor:int=None, 
                 learning_head:int=None) -> None:
        super().__init__(observation_space, features_dim)
        
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dim_flatten = self.flatten(torch.as_tensor(observation_space.sample()[None]).float()).shape[-1]
            
        self.mlp_layers = nn.Sequential(
            nn.Linear(dim_flatten, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        return self.mlp_layers(self.flatten(observations))