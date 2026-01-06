from __future__ import annotations
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import TYPE_CHECKING, Optional
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
import torch
import torch.nn as nn

if TYPE_CHECKING:
    import gymnasium as gym

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)

    def forward(self, x):
        inputs = x
        x = torch.relu(x)
        x = self.conv0(x)
        x = torch.relu(x)
        x = self.conv1(x)
        return x + inputs

class ImpalaCNNLarge(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        normalized_image: bool = False
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "ImpalaCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )

        n_input_channels = observation_space.shape[0]

        super().__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use ImpalaCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        

        # Modified CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            ResidualBlock(channels= 32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            ResidualBlock(channels= 64),
            nn.Flatten()
        )

        # Calculate flattened size
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        self.fc = nn.Linear(n_flatten, features_dim)
    

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # Pixel normalization (/255)
        x = x / 255.0

        # Forward pass through the Sequential block
        x = self.cnn(x)

        # Pass flattened output through the fully-connected layer
        x = self.fc(x)

        return x
    
class ImpalaCNNSmall(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "ImpalaCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use ImpalaCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )

        # Modified CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate flattened size
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        self.fc = nn.Linear(n_flatten, features_dim)



    def forward(self, x:torch.Tensor)->torch.Tensor:
        # Pixel normalization (/255)
        x = x / 255.0
        
        # Forward pass through the Sequential block
        x = self.cnn(x)

        # Pass flattened output through the fully-connected layer
        x = self.fc(x)

        return x