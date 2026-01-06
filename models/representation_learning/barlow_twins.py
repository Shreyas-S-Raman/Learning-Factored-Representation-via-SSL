from __future__ import annotations
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import gymnasium as gym
from models.encoder_layers.impala_cnn import ImpalaCNNLarge, ImpalaCNNSmall
from models.encoder_layers.nature_cnn import NatureCNN
from models.representation_learning_utils.barlow_twins import LARS
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import LambdaLR
    
class BarlowTwinsRepresentationLearner(BaseFeaturesExtractor):

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
        super(BarlowTwinsRepresentationLearner).__init__(
            observation_space, 
            representation_vector.vector_size_per_factor*representation_vector.num_factors
        )
        # define and create visual encoder
        observation_encoders = {
            'impala-large': ImpalaCNNLarge,
            'impala-small': ImpalaCNNSmall,
            'nature': NatureCNN
        }
        self.observation_encoder = observation_encoders[obs_encoder.encoder](
            observation_space=observation_space,
            features_dim=observation_encoder_dim,
            normalized_image=normalized_image
        )
        layers = []
        for i in range(0, len(projection_architecture)):
            layers.append(nn.ReLU())
            layers.append(
                nn.Linear(projection_architecture[i-1], projection_architecture[i])
            )
        layers.append(projection_architecture[-1], representation_vector.vector_size_per_factor*representation_vector.num_factors)
        self.linear_projection = nn.Sequential(*layers)
        
        # if needed define dreamer v2 style RSSM
        if obs_encoder.enable_rssm:
            raise NotImplementedError("to be implemented with DV2 updates")

        self.transform = T.Compose([
            T.RandomResizedCrop(
                size=observation_space.shape,
                scale=(0.08, 1.0),      # BT uses the BYOL values
                ratio=(3./4., 4./3.),   # BYOL/SimCLR aspect-ratio range
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.RandomApply([
                T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1,
                )
            ], p=0.8),
            T.RandomApply([
                T.GaussianBlur(
                    kernel_size=int(0.1 * observation_space.shape) | 1,    # ensure odd
                    sigma=(0.1, 2.0)
                )
            ], p=0.1)
        ])
        self.lambda_offdiag = 0.005
        self.optimizer_params = optimizer_params
    
    def forward(self, x:torch.Tensor, actions:torch.Tensor=None, test:bool=True)->torch.Tensor:
        if not test:
            x_transformed = []
            for i in range(x.shape[0]):
                aug = self.transform(x[i].permute(2, 0, 1))
                x_transformed.append(aug)
            x = torch.stack(x_transformed, dim=0)
        x = self.observation_encoder(x)
        x = self.linear_projection(x)
        if test:
            x = x.detach()
        return x
    
    def off_diagonal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return a flattened view of the off-diagonal elements of a square matrix.
        """
        n, m = x.shape
        assert n == m, "Input to off_diagonal must be a square matrix"
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def compute_loss(self, batch_dictionary):
        # convert to tensors
        observations = torch.as_tensor(batch_dictionary.observations).float().to(self.device)
        # forward pass
        with torch.set_grad_enabled(True):
            z_a = self.forward(observations, test=False)
            z_b = self.forward(observations, test=False)
            N, D = z_a.shape

            # Normalize across batch dim per feature
            z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + self.eps)
            z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + self.eps)

            # Cross-correlation matrix
            c = (z_a_norm.T @ z_b_norm) / N  # (D, D)

            # On-diagonal should be 1
            on_diag = torch.diagonal(c)
            on_diag_loss = ((on_diag - 1) ** 2).sum()

            # Off-diagonals should be 0
            off_diag_vals = self.off_diagonal(c)
            off_diag_loss = (off_diag_vals ** 2).sum()

            loss = on_diag_loss + self.lambda_offdiag * off_diag_loss
        return loss, {'cross_correlation': (c, True), 'on_diag': (on_diag, False), 'on_diag_loss': (on_diag_loss, False), 'off_diag_loss': (off_diag_loss, False)}

    def build_optimizers(self):
        # lr scaling used in paper
        lr = self.optimizer_params["lr"] * (self.optimizer_params["batch_size"] / 256.0)
        sgd = torch.optim.SGD(
            self.parameters(),
            lr=lr,
            momentum= self.optimizer_params.get("momentum", 0.9),
            weight_decay= self.optimizer_params.get("weight_decay", 1.5e-6),
        )
        opt = LARS(
            optimizer=sgd,
            trust_coefficient= self.optimizer_params.get("trust_coefficient", 0.001),
            eps= self.optimizer_params.get("eps", 1e-9),
            clip= self.optimizer_params.get("clip", False),
        )
        return {"all": opt}

    def setup_schedules(self, optimizer):
        cfg = self.optimizer_params
        warmup_steps = int(cfg.get("warmup_scheduler_steps", 0))
        total_steps = int(cfg["total_scheduler_steps"])
        final_scale = float(cfg.get("final_lr_scale", 0.001))  # 1/1000

        def lr_lambda(step: int):
            # --- linear warmup ---
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps

            # --- cosine decay ---
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)

            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return final_scale + (1 - final_scale) * cosine

        return LambdaLR(optimizer, lr_lambda)
        
    def post_step(self):
        return {}






