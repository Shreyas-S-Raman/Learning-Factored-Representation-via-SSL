from models.encoder_layers.impala_cnn import ImpalaCNNLarge, ImpalaCNNSmall
from models.encoder_layers.nature_cnn import NatureCNN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import TYPE_CHECKING
import torch.nn as nn
from copy import copy
import torch
if TYPE_CHECKING:
    import gymnasium as gym
    

class CURLRepresentationLearner(BaseFeaturesExtractor):

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
        super(CURLRepresentationLearner).__init__(
            observation_space, 
            representation_vector.vector_size_per_factor*representation_vector.num_factors
        )
        # define and create visual encoder
        observation_encoders = {
            'impala-large': ImpalaCNNLarge,
            'impala-small': ImpalaCNNSmall,
            'nature': NatureCNN
        }
        self.observation_key_encoder = observation_encoders[obs_encoder.encoder]
        self.observation_key_encoder(
            observation_space=observation_space,
            features_dim=observation_encoder_dim,
            normalized_image=normalized_image
        )

        self.observation_query_encoder = observation_encoders[obs_encoder.encoder]
        self.observation_query_encoder(
            observation_space=observation_space,
            features_dim=observation_encoder_dim,
            normalized_image=normalized_image
        )

        # if needed define dreamer v2 style RSSM
        if obs_encoder.enable_rssm:
            raise NotImplementedError("to be implemented with DV2 updates")
        
        # include projection layers to get final latent representation in init
        layers = [nn.Linear(observation_encoder_dim, projection_architecture[0])]
        for i in range(1, len(projection_architecture)):
            layers.append(nn.ReLU())
            layers.append(
                nn.Linear(projection_architecture[i-1], projection_architecture[i])
            )
        layers.append(nn.Linear(projection_architecture[-1], representation_vector.vector_size_per_factor*representation_vector.num_factors))
        self.key_proj = nn.Sequential(*layers)
        self.query_proj = nn.Sequential(copy(*layers))

        # bilinear product weights
        self.W = nn.Linear(
            representation_vector.vector_size_per_factor*representation_vector.num_factors,
            representation_vector.vector_size_per_factor*representation_vector.num_factors
        )

        # define additional variables for auxiliary objective
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer_params = optimizer_params
    
    def curl_crop(self, x, output_size=84):
        # x: (B, C, H, W)
        _, _, H, W = x.shape

        # This is built-in: returns random (top, left, height, width)
        i, j, h, w = T.RandomCrop.get_params(
            img=torch.zeros(H, W),  # dummy, only shape matters
            output_size=(output_size, output_size),
        )

        # Apply SAME crop to all images
        return x[:, :, i:i+h, j:j+w]

    def forward(self, x:torch.Tensor, actions:torch.Tensor=None, test:bool=True)->torch.Tensor:
        x = self.observation_encoder(x)
        if test:
            x = self.query_proj(self.observation_query_encoder(x))
        # only during training time: crop images and perform bilinear product to get logits & labels
        else:
            x = self.key_proj(self.observation_key_encoder(x)).detach()
        return x
    
    def compute_loss(self, batch_dictionary):
        # convert to tensors
        observations = torch.as_tensor(batch_dictionary.observations).float().to(self.device)
        obs_query = self.curl_crop(observations)
        obs_key = self.curl_crop(observations)
        # forward pass
        with torch.set_grad_enabled(True):
            z_query = self.forward(obs_query, test=True)
            z_key = self.forward(obs_key, test=False)
            logits = torch.matmul(z_query, torch.matmul(self.W, z_key.T))
            logits = logits - torch.max(logits, axis=1)
            labels = torch.arange(logits.shape[0])
            loss = self.loss_fn(logits, labels)
            accuracy = torch.sum(logits==labels)/logits.shape[0]
        return loss, {'accuracy': (accuracy, False), 'W': (self.W.weight.cpu(), True)}
    
    def build_optimizers(self):
        fq_params = list(self.observation_query_encoder.parameters()) \
              + list(self.query_proj.parameters())
        
        return { 
            'query_function': torch.optim.Adam(
                fq_params,
                **self.optimizer_params
            ),
            'W': torch.optim.Adam(
                self.W.parameters(),
                **self.optimizer_params
            )
        }