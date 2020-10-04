from typing import Optional, Tuple, Type

import torch

from torch import Tensor, nn
from torch.distributions import Categorical

from rainy.net import (
    CNNBody,
    FCBody,
    Initializer,
    LinearHead,
    NetworkBlock,
    OptionCriticNet,
    PolicyDist,
    SharedBodyOCNet,
    policy_init,
)
from rainy.net.policy import BernoulliDist, BernoulliPolicy, CategoricalDist, Policy
from rainy.net.prelude import NetFn
from rainy.prelude import ArrayLike
from rainy.utils import Device


class PandMuMixIn:
    """ Represents p^ and μ^
    """

    device: Device
    pmu_body: nn.Module
    _po_xs: nn.Module
    _po_xsxf: nn.Module

    def muo_xs(self, xs: ArrayLike) -> Categorical:
        """ Returns μ^(・|xs).
        """
        xst = self.device.tensor(xs)
        xs_feature = self.pmu_body(xst)
        xs_logits = self._po_xs(xs_feature)
        return Categorical(logits=xs_logits)

    def po_xsxf(self, xs: ArrayLike, xf: ArrayLike) -> Categorical:
        """ Returns p^(・|xs, xf).
        """
        xst, xft = self.device.tensor(xs), self.device.tensor(xf)
        xs_feature, xf_feature = self.pmu_body(xst), self.pmu_body(xft)
        xs_xf = torch.cat((xs_feature, xf_feature), dim=1)
        xs_xf_logits = self._po_xsxf(xs_xf)
        return Categorical(logits=xs_xf_logits)

    def muo_xs_po_xsxf(
        self, xs: ArrayLike, xf: ArrayLike,
    ) -> Tuple[Categorical, Categorical]:
        """ Returns μ^(・|xs) and p^(・|xs, xf).
        """
        xst, xft = self.device.tensor(xs), self.device.tensor(xf)
        xs_feature, xf_feature = self.pmu_body(xst), self.pmu_body(xft)
        xs_xf = torch.cat((xs_feature, xf_feature), dim=1)
        xs_logits = self._po_xs(xs_feature)
        xs_xf_logits = self._po_xsxf(xs_xf)
        return Categorical(logits=xs_logits), Categorical(logits=xs_xf_logits)

    def muo_xf_po_xsxf(
        self, xs: ArrayLike, xf: ArrayLike,
    ) -> Tuple[Categorical, Categorical]:
        """ Returns μ^(・|xf) and p^(・|xs, xf).
            Only for visualization.
        """
        xst, xft = self.device.tensor(xs), self.device.tensor(xf)
        xs_feature, xf_feature = self.pmu_body(xst), self.pmu_body(xft)
        xs_xf = torch.cat((xs_feature, xf_feature), dim=1)
        xf_logits = self._po_xs(xf_feature)
        xs_xf_logits = self._po_xsxf(xs_xf)
        return Categorical(logits=xf_logits), Categorical(logits=xs_xf_logits)


class ImocNet(SharedBodyOCNet, PandMuMixIn):
    def __init__(
        self,
        body: NetworkBlock,
        action_dim: int,
        num_options: int,
        policy_dist: PolicyDist,
        init: Initializer = Initializer(),
        beta_init: Optional[Initializer] = None,
        policy_init: Initializer = policy_init(),
        device: Device = Device(),
    ) -> None:
        super().__init__(
            body,
            action_dim,
            num_options,
            policy_dist,
            init=init,
            beta_init=beta_init,
            policy_init=policy_init,
            device=device,
        )
        out_dim = self.body.output_dim
        self._po_xs = LinearHead(out_dim, self.num_options, init=init)
        self._po_xsxf = init(
            nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, self.num_options),
            )
        )
        self.pmu_body = self.body
        self.to(device.unwrapped)

    def pqbm(
        self, states: ArrayLike,
    ) -> Tuple[Policy, Tensor, BernoulliPolicy, Categorical]:
        """ Returns π
        μ^(・|xf) and p^(・|xs, xf).
            Only for visualization.
        """
        feature = self.body(self.device.tensor(states))
        policy = self.actor_head(feature).view(-1, self.num_options, self.action_dim)
        qo = self.qo_head(feature)
        beta = self.beta_head(feature)
        po_xs = Categorical(logits=self._po_xs(feature))
        return self.policy_dist(policy), qo, self.beta_dist(beta), po_xs

    def forward(self, states: ArrayLike) -> Tuple[Policy, Tensor, BernoulliPolicy]:
        feature = self.body(self.device.tensor(states))
        policy = self.actor_head(feature).view(-1, self.num_options, self.action_dim)
        qo = self.qo_head(feature)
        beta = self.beta_head(feature)
        return self.policy_dist(policy), qo, self.beta_dist(beta)


class ImocRndNet(ImocNet):
    def __init__(self, *args, init: Initializer = Initializer(), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.int_qo_head = LinearHead(
            self.body.output_dim, self.num_options, init=init,
        )
        self.int_qo_head.to(self.device.unwrapped)

    def pqqbm(
        self, states: ArrayLike
    ) -> Tuple[Policy, Tensor, Tensor, BernoulliPolicy, Categorical]:
        feature = self.body(self.device.tensor(states))
        policy = self.actor_head(feature).view(-1, self.num_options, self.action_dim)
        qo = self.qo_head(feature)
        int_qo = self.int_qo_head(feature)
        beta = self.beta_head(feature)
        po_xs = Categorical(logits=self._po_xs(feature))
        return self.policy_dist(policy), qo, int_qo, self.beta_dist(beta), po_xs

    def forward(
        self, states: ArrayLike,
    ) -> Tuple[Policy, Tensor, Tensor, BernoulliPolicy]:
        feature = self.body(self.device.tensor(states))
        policy = self.actor_head(feature).view(-1, self.num_options, self.action_dim)
        qo = self.qo_head(feature)
        int_qo = self.int_qo_head(feature)
        beta = self.beta_head(feature)
        return self.policy_dist(policy), qo, int_qo, self.beta_dist(beta)


class SeparatedImocNet(OptionCriticNet, PandMuMixIn):
    """ Similar to ImocNet, but has a separated bodies for p^, μ^, and β
    """

    def __init__(
        self,
        body: NetworkBlock,
        pmu_body: NetworkBlock,
        action_dim: int,
        num_options: int,
        policy_dist: PolicyDist,
        init: Initializer = Initializer(),
        beta_init: Optional[Initializer] = None,
        policy_init: Initializer = policy_init(),
        device: Device = Device(),
    ) -> None:
        super().__init__()
        self.has_mu = False
        self.body = body
        self.pmu_body = pmu_body
        self.actor_head = LinearHead(
            body.output_dim, num_options * action_dim, init=policy_init
        )
        self.qo_head = LinearHead(body.output_dim, num_options, init=init)
        self.beta_head = LinearHead(
            body.output_dim, num_options, init=beta_init or init
        )
        self.policy_dist = policy_dist
        self.beta_dist = BernoulliDist()
        self.num_options = num_options
        self.action_dim = action_dim
        self.device = device
        self.state_dim = self.body.input_dim
        # PandMuMixIn
        out_dim = self.body.output_dim
        self._po_xs = LinearHead(out_dim, self.num_options, init=init)
        self._po_xsxf = init(
            nn.Sequential(
                nn.Linear(out_dim * 2, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, self.num_options),
            )
        )
        self.to(device.unwrapped)

    def qo(self, states: ArrayLike) -> Tensor:
        feature = self.body(self.device.tensor(states))
        return self.qo_head(feature)

    def beta(self, states: ArrayLike) -> BernoulliPolicy:
        feature = self.pmu_body(self.device.tensor(states))
        return self.beta_dist(self.beta_head(feature))

    def qo_and_beta(self, states: ArrayLike) -> Tuple[Tensor, BernoulliPolicy]:
        feature = self.body(self.device.tensor(states))
        beta_feature = self.pmu_body(self.device.tensor(states))
        return self.qo_head(feature), self.beta_dist(self.beta_head(beta_feature))

    def pqbm(
        self, states: ArrayLike,
    ) -> Tuple[Policy, Tensor, BernoulliPolicy, Categorical]:
        feature = self.body(self.device.tensor(states))
        beta_feature = self.pmu_body(self.device.tensor(states))
        policy = self.actor_head(feature).view(-1, self.num_options, self.action_dim)
        qo = self.qo_head(feature)
        beta = self.beta_head(beta_feature)
        po_xs = Categorical(logits=self._po_xs(beta_feature))
        return self.policy_dist(policy), qo, self.beta_dist(beta), po_xs

    def forward(self, states: ArrayLike) -> Tuple[Policy, Tensor, BernoulliPolicy]:
        feature = self.body(self.device.tensor(states))
        beta_feature = self.pmu_body(self.device.tensor(states))
        policy = self.actor_head(feature).view(-1, self.num_options, self.action_dim)
        qo = self.qo_head(feature)
        beta = self.beta_head(beta_feature)
        return self.policy_dist(policy), qo, self.beta_dist(beta)


def conv_shared(
    num_options: int = 8,
    policy: Type[PolicyDist] = CategoricalDist,
    hidden_channels: Tuple[int, int, int] = (32, 64, 32),
    feature_dim: int = 256,
    beta_init: Initializer = Initializer(),
    cls: Type[OptionCriticNet] = ImocNet,
    **kwargs,
) -> NetFn:
    def _net(
        state_dim: Tuple[int, int, int], action_dim: int, device: Device
    ) -> OptionCriticNet:
        body = CNNBody(
            state_dim, hidden_channels=hidden_channels, output_dim=feature_dim, **kwargs
        )
        dist = policy(action_dim, device, noptions=num_options)
        return cls(
            body, action_dim, num_options, dist, beta_init=beta_init, device=device,
        )

    return _net  # type: ignore


def fc_shared(
    num_options: int = 8,
    policy: Type[PolicyDist] = CategoricalDist,
    has_mu: bool = False,
    beta_init: Initializer = Initializer(),
    cls: Type[OptionCriticNet] = ImocNet,
    **kwargs,
) -> NetFn:
    def _net(state_dim: Tuple[int], action_dim: int, device: Device) -> SharedBodyOCNet:
        body = FCBody(state_dim[0], **kwargs)
        dist = policy(action_dim, device, noptions=num_options)
        return cls(
            body, action_dim, num_options, dist, beta_init=beta_init, device=device,
        )

    return _net  # type: ignore


def fc_separated(
    num_options: int = 8,
    policy: Type[PolicyDist] = CategoricalDist,
    has_mu: bool = False,
    beta_init: Initializer = Initializer(),
    cls: Type[OptionCriticNet] = SeparatedImocNet,
    **kwargs,
) -> NetFn:
    def _net(state_dim: Tuple[int], action_dim: int, device: Device) -> SharedBodyOCNet:
        body1 = FCBody(state_dim[0], **kwargs)
        body2 = FCBody(state_dim[0], **kwargs)
        dist = policy(action_dim, device, noptions=num_options)
        return cls(
            body1,
            body2,
            action_dim,
            num_options,
            dist,
            beta_init=beta_init,
            device=device,
        )

    return _net  # type: ignore
