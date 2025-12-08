from models.encoder_layers.impala_cnn import ImpalaCNNLarge, ImpalaCNNSmall
from models.encoder_layers.nature_cnn import NatureCNN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import gymnasium as gym
    import torch

class VisualRepresentationLearner(BaseFeaturesExtractor):

    def __init__(self,
        observation_space: gym.Space,
        obs_encoder: dict,
        representation_vector: dict,
        projection_architecture: list,
        rssm_configs: dict,
        observation_encoder_dim: int = 256,
        expert_obs: gym.Space= None,
        num_actions: int=3,
        normalized_image: bool = False):
        super(VisualRepresentationLearner).__init__(
            observation_space, 
            representation_vector.vector_size_per_factor*representation_vector.num_factors
        )
        # define and create visual encoder
        observation_encoders = {
            'impala-large': ImpalaCNNLarge,
            'impala-small': ImpalaCNNSmall,
            'nature': NatureCNN
        }
        self.observation_encoder = observation_encoders[obs_encoder.encoder]
        self.observation_encoder(
            observation_space=observation_space,
            features_dim=observation_encoder_dim,
            normalized_image=normalized_image
        )
        layers = [nn.Linear(observation_encoder_dim, projection_architecture[0])]
        for i in range(1, len(projection_architecture)):
            layers.append(nn.ReLU())
            layers.append(
                nn.Linear(projection_architecture[i-1], projection_architecture[i])
            )
        layers.append(projection_architecture[-1], representation_vector.vector_size_per_factor*representation_vector.num_factors)
        self.linear_projection = nn.Sequential(*layers)

        # if needed define dreamer v2 style RSSM
        if obs_encoder.enable_rssm:
            raise NotImplementedError("to be implemented with DV2 updates")
        
    def forward(self, x:torch.Tensor, actions:torch.Tensor=None, test:bool=True)->torch.Tensor:
        x = self.observation_encoder(x)
        x = self.linear_projection(x)
        return x
