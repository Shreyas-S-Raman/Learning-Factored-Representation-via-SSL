from __future__ import annotations
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import gymnasium as gym
class FactoredVFLearner(BaseFeaturesExtractor):

    def __init__(self,
        observation_space: gym.Space,
        obs_encoder: dict,
        representation_vector: dict,
        projection_architecture: list,
        rssm_configs: dict,
        optimizer_params: dict,
        observation_encoder_dim: int = 256,
        expert_obs: gym.Space= None,
        num_actions: int=3,
        normalized_image: bool = False):
        return None
    
