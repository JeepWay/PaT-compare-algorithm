from typing import Any, Dict, List, Optional, Type, Tuple, ClassVar, Union

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
# from stable_baselines3.common.torch_layers import (
#     BaseFeaturesExtractor,
#     CombinedExtractor,
#     FlattenExtractor,
#     NatureCNN,
#     create_mlp,
# )
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from deep_pack.common.preprocessing import preprocess_obs
from deep_pack.common.torch_layers import (
    BaseNetwork,
    CnnMlpNetwork1, 
    CnnMlpNetwork2, 
    CnnMlpNetwork3, 
    CnnMlpNetwork4,
)
from deep_pack.common.constants import BIN, MASK

class QNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        # features_extractor: BaseFeaturesExtractor,
        # features_dim: int,
        # net_arch: Optional[List[int]] = None,
        # activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        network_class: Type[BaseNetwork] = CnnMlpNetwork1,
        network_kwargs: Optional[Dict[str, Any]] = None,
        mask_replace_coef: int = -15,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            # features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        # if net_arch is None:
        #     net_arch = [64, 64]

        # self.net_arch = net_arch
        # self.activation_fn = activation_fn
        # self.features_dim = features_dim
        # action_dim = int(self.action_space.n)  # number of actions
        # q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        # self.q_net = nn.Sequential(*q_net)

        self.network_class = network_class
        self.network_kwargs = network_kwargs
        self.mask_replace_coef = mask_replace_coef

        self.q_net = self.network_class(
            observation_space=self.observation_space[BIN], 
            action_dim=int(self.action_space.n), 
            **self.network_kwargs,
        )

    def forward(self, obs: PyTorchObs) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.s
        """
        obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.q_net(obs[BIN])

    def _predict(self, observation: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        masked_q_values = th.where(observation[MASK] == 1, q_values, self.mask_replace_coef)
        # Greedy action
        # action = q_values.argmax(dim=1).reshape(-1)
        action = masked_q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                # net_arch=self.net_arch,
                # features_dim=self.features_dim,
                # activation_fn=self.activation_fn,
                # features_extractor=self.features_extractor,
                network_class=self.network_class,
                network_kwargs=self.network_kwargs,
                mask_replace_coef=self.mask_replace_coef,
            )
        )
        return data


class DQNPolicy(BasePolicy):
    """
    Policy class for DQN when using images as input.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    q_net: QNetwork
    q_net_target: QNetwork

    network_aliases: ClassVar[Dict[str, Type[BaseNetwork]]] = {
        "CnnMlpNetwork1": CnnMlpNetwork1,
        "CnnMlpNetwork2": CnnMlpNetwork2,
        "CnnMlpNetwork3": CnnMlpNetwork3,
        "CnnMlpNetwork4": CnnMlpNetwork4,
    }

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        # net_arch: Optional[List[int]] = None,
        # activation_fn: Type[nn.Module] = nn.ReLU,
        # features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        # features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = False,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        network: Union[str, Type[BaseNetwork]] = CnnMlpNetwork1,
        network_kwargs: Optional[Dict[str, Any]] = None,
        mask_replace_coef: int = -15,
    ) -> None:
        if isinstance(network, str):
            self.network_class = self._get_network_from_name(network)
        else:
            self.network_class = network
        self.network_kwargs = network_kwargs
        self.mask_replace_coef = mask_replace_coef

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            # features_extractor_class,
            # features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            # "net_arch": self.net_arch,
            # "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "network_class": self.network_class,
            "network_kwargs": self.network_kwargs,
            "mask_replace_coef": self.mask_replace_coef,
        }

        self._build(lr_schedule)

    def _get_network_from_name(self, network_name: str) -> Type[BaseNetwork]:
        """
        Get a network class from its name representation.

        :param network_name: Alias of the network
        :return: A network class (type)
        """
        if network_name in self.network_aliases:
            return self.network_aliases[network_name]
        else:
            raise ValueError(f"Policy {network_name} unknown")

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.

        Put the target network into evaluation mode.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.q_net.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def make_q_net(self) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        return QNetwork(**self.net_args).to(self.device)

    def forward(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: PyTorchObs, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                # net_arch=self.net_args["net_arch"],
                # activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                # features_extractor_class=self.features_extractor_class,
                # features_extractor_kwargs=self.features_extractor_kwargs,
                network_class=self.network_class,
                network_kwargs=self.network_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net.set_training_mode(mode)
        self.training = mode


CnnMlpPolicy = DQNPolicy