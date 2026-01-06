from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, TYPE_CHECKING
if TYPE_CHECKING:
    import gymnasium as gym

class SupervisedLearningHead(nn.Module):

    def __init__(self, observation_encoder_dim: int, vector_size_per_factor:int, num_factors:int, expert_obs:gym.Space, projection_architecture:list):
        super(SupervisedLearningHead, self).__init__()

        layers = []
        for i in range(0, len(projection_architecture)):
            layers.append(nn.ReLU())
            from_dim = observation_encoder_dim if i==0 else projection_architecture[i-1]
            layers.append(
                nn.Linear(from_dim, projection_architecture[i])
            )
        layers.append(nn.ReLU())
        layers.append(nn.Linear(
            observation_encoder_dim if i==0 else projection_architecture[-1],
            num_factors*vector_size_per_factor
        ))
        self.expert_scale_proj = nn.Sequential(*layers)
        
        self.expert_distribution_proj = nn.ModuleList([nn.Linear(vector_size_per_factor, 1 if isinstance(space, float) else space.high) for (key, space) in expert_obs.spaces.items()])
        self.softmax = nn.Softmax(dim=-1)
        self.discrete_state = [False if isinstance(space, float) else True]

        #extra variables to reshape output
        self.num_factors = num_factors
        self.vector_size_per_factor = vector_size_per_factor
    
    def forward(self, x: torch.Tensor, test:bool=True)->torch.Tensor:
        
        #if in eval mode: used for PPO policy learning => then detach computation and take argmax
        if test:
            x = self.expert_scale_proj(x)
            x = x.reshape(x.shape[0], self.num_factors, self.vector_size_per_factor)
            x = [torch.argmax(fc_proj(x[:,i,:]), axis=-1)  if self.discrete_state[i] else fc_proj(x[:,i,:]) for i, fc_proj in enumerate(self.expert_distribution_proj)]
            
        #if in train mode: used for SL representation learning  ==> then do not detach computation and do not take argmax
        else:
            x = self.expert_scale_proj(x)
            x = x.reshape(x.shape[0], self.num_factors, self.vector_size_per_factor)
            x = [self.softmax(fc_proj(x[:,i,:])) if self.discrete_state[i] else fc_proj(x[:,i,:]) for i, fc_proj in enumerate(self.expert_distribution_proj)]
        
        return x




