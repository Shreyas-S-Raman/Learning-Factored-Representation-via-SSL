from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import gymnasium as gym
    import torch

class ExpertRepresentationLearner(BaseFeaturesExtractor):

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
        super(ExpertRepresentationLearner).__init__(
            observation_space, 
            representation_vector.vector_size_per_factor*representation_vector.num_factors
        )
        

        # if needed define dreamer v2 style RSSM
        if obs_encoder.enable_rssm:
            raise NotImplementedError("to be implemented with DV2 updates")
        
    def forward(self, x:torch.Tensor, actions:torch.Tensor=None, test:bool=True)->torch.Tensor:
        # NOTE: we do not apply any transformation or mapping observations, expert representation directly passed to PPO
        return x

    def post_step(self):
        return {}