import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

class BaseNetwork(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int = 100,
        normalize_images: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "BaseNetwork must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__()
        self.observation_space = observation_space
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.normalize_images = normalize_images
        self.input_channels = observation_space.shape[0]

        # to be defined in the subclasses 
        self.out_channels: int = None
        self.cnn_extractor: nn.Sequential = None
        self.n_flatten: int = None
        self.mlp_extractor: nn.Sequential = None
        
    def _get_n_flatten(self) -> int:
        # Compute shape after flattening by doing one forward pass
        with th.no_grad():
            obs_tenosr = th.as_tensor(self.observation_space.sample()[None]).float()
            n_flatten = self.cnn_extractor(obs_tenosr).shape[1]
        return n_flatten

    def forward(self, observations: th.Tensor) -> th.Tensor:
        cnn_features = self.cnn_extractor(observations)
        return self.mlp_extractor(cnn_features)


class CnnMlpNetwork1(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int = 200,
        normalize_images: bool = False,
        out_channels: int = 8,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
        )
        self.out_channels = out_channels
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2,2)),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2,2)),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2,2)),
            nn.Conv2d(16, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.n_flatten = self._get_n_flatten()
        self.mlp_extractor = nn.Sequential(
            nn.Linear(self.n_flatten, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )


class CnnMlpNetwork2(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int = 400,
        normalize_images: bool = False,
        out_channels: int = 8,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
        )
        self.out_channels = out_channels
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2,2)),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2,2)),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2,2)),
            nn.Conv2d(16, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.n_flatten = self._get_n_flatten()
        self.mlp_extractor = nn.Sequential(
            nn.Linear(self.n_flatten, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )


class CnnMlpNetwork3(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int = 1600,
        normalize_images: bool = False,
        out_channels: int = 8,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
        )
        self.out_channels = out_channels
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2,2)),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2,2)),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2,2)),
            nn.Conv2d(16, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.n_flatten = self._get_n_flatten()
        self.mlp_extractor = nn.Sequential(
            nn.Linear(self.n_flatten, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )


class CnnMlpNetwork4(BaseNetwork):
    def __init__(
        self,
        observation_space: gym.Space,
        action_dim: int,
        hidden_dim: int = 1600,
        normalize_images: bool = False,
        out_channels: int = 8,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            normalize_images=normalize_images,
        )
        self.out_channels = out_channels
        self.cnn_extractor = nn.Sequential(
            nn.Conv2d(self.input_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2,2)),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2,2)),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=(2,2)),
            nn.Conv2d(16, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.n_flatten = self._get_n_flatten()
        self.mlp_extractor = nn.Sequential(
            nn.Linear(self.n_flatten, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
