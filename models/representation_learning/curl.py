from models.encoder_layers.impala_cnn import ImpalaCNNLarge, ImpalaCNNSmall
from models.encoder_layers.nature_cnn import NatureCNN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import TYPE_CHECKING
import torch.nn as nn
import copy
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
        self.observation_key_encoder = observation_encoders[obs_encoder.encoder](
            observation_space=observation_space,
            features_dim=observation_encoder_dim,
            normalized_image=normalized_image
        )

        self.observation_query_encoder = observation_encoders[obs_encoder.encoder](
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
        self.query_proj = nn.Sequential(*layers)
        self.key_proj = copy.deepcopy(self.query_proj)
        for p in self.observation_key_encoder.parameters():
            p.requires_grad = False
        for p in self.key_proj.parameters():
            p.requires_grad = False

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
        if test:
            x = self.query_proj(self.observation_query_encoder(x)).detach()
        # only during training time: crop images and perform bilinear product to get logits & labels
        else:
            x = self.query_proj(self.observation_query_encoder(x))
        return x
    
    def forward_key(self, x:torch.Tensor, actions:torch.Tensor=None, test:bool=True)->torch.Tensor:
        x = self.key_proj(self.observation_key_encoder(x)).detach()
        return x

    def compute_loss(self, batch_dictionary):
        # convert to tensors
        observations = torch.as_tensor(batch_dictionary.observations).float().to(self.device)
        obs_query = self.curl_crop(observations)
        obs_key = self.curl_crop(observations)

        # forward pass
        with torch.set_grad_enabled(True):
            z_query = self.forward(obs_query, test=False)
            z_key = self.forward_key(obs_key)
            logits = torch.matmul(z_query, torch.matmul(self.W, z_key.T))
            logits = logits - logits.max(dim=1, keepdim=True)[0]
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss = self.loss_fn(logits, labels)
            preds = logits.argmax(dim=1)
            accuracy = torch.sum(preds==labels)/logits.shape[0]
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
    
    def post_step(self):
        self.momentum_update_key()
        return {}

    @torch.no_grad()
    def momentum_update_key(self, momentum: float = 0.99):
        """
        EMA update:
            theta_k = m * theta_k + (1 - m) * theta_q
        Applied to both encoder and projection head.
        """
        momentum = self.optimizer_params.get('momentum', momentum)
        # encoders
        for p_k, p_q in zip(
            self.observation_key_encoder.parameters(),
            self.observation_query_encoder.parameters()
        ):
            p_k.data.mul_(momentum).add_(p_q.data, alpha=1.0 - momentum)

        # projection heads
        for p_k, p_q in zip(self.key_proj.parameters(), self.query_proj.parameters()):
            p_k.data.mul_(momentum).add_(p_q.data, alpha=1.0 - momentum)