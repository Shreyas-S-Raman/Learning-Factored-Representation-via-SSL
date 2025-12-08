from models.encoder_layers.impala_cnn import ImpalaCNNLarge, ImpalaCNNSmall
from models.encoder_layers.nature_cnn import NatureCNN
from models.representation_learning_utils.supervised import SupervisedLearningHead
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import gymnasium as gym
    import torch

class SupervisedRepresentationLearner(BaseFeaturesExtractor):

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
        super(SupervisedRepresentationLearner).__init__(
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

        # if needed define dreamer v2 style RSSM
        if obs_encoder.enable_rssm:
            raise NotImplementedError("to be implemented with DV2 updates")
        
        # define supervised learning head
        self.supervised_learning_head = SupervisedLearningHead(
            observation_encoder_dim = observation_encoder_dim, 
            vector_size_per_factor = representation_vector.vector_size_per_factor,
            num_factors = num_factors, 
            expert_obs = expert_obs,
            projection_architecture = projection_architecture
        )
        
    def forward(self, x:torch.Tensor, actions:torch.Tensor=None, test:bool=True)->torch.Tensor:
        x = self.observation_encoder(x)
        x = self.supervised_learning_head(x, test=test)
        return x
