""" PPIMOC Implementation
"""


from typing import NamedTuple, Sequence

import numpy as np
import torch

from torch import Tensor

from a2imoc import A2ImocAgent

# Local modules
from rainy.agents import PPOLossMixIn
from rainy.config import Config
from rainy.envs import ParallelEnv
from rainy.lib import rollout
from rainy.prelude import Array, Index, State
from rollout import RolloutStorage


class RolloutBatch(NamedTuple):
    states: Tensor
    actions: Tensor
    returns: Tensor
    old_log_probs: Tensor
    old_betas: Tensor
    old_beta_logits: Tensor
    muo_xses: Tensor
    advantages: Tensor
    beta_advs: Tensor
    options: Tensor
    prev_options: Tensor


class RolloutSampler(rollout.RolloutSampler):
    def __init__(
        self,
        storage: RolloutStorage,
        penv: ParallelEnv,
        config: Config,
        beta_adv: Tensor,
        muo_xses: Tensor,
        options: Tensor,
        prev_options: Tensor,
        batch_indices: Tensor,
    ) -> None:
        torch.stack(storage.values, out=storage.batch_values)
        super().__init__(
            storage, penv, config.ppo_minibatch_size, config.adv_normalize_eps
        )
        self.options = options
        self.prev_options = prev_options
        self.beta_advs = beta_adv
        self.muo_xses = muo_xses
        betas, beta_logits = torch.cat(storage.betas), torch.cat(storage.beta_logits)
        self.old_betas = betas[batch_indices, prev_options]
        self.old_beta_logits = beta_logits[batch_indices, prev_options]

    def _make_batch(self, i: Index) -> RolloutBatch:
        return RolloutBatch(
            self.states[i],
            self.actions[i],
            self.returns[i],
            self.old_log_probs[i],
            self.old_betas[i],
            self.old_beta_logits[i],
            self.muo_xses[i],
            self.advantages[i],
            self.beta_advs[i],
            self.options[i],
            self.prev_options[i],
        )


class PPImocAgent(A2ImocAgent, PPOLossMixIn):
    SAVED_MEMBERS = "net", "optimizer"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.clip_cooler = config.clip_cooler()
        self.clip_eps = config.ppo_clip
        self.num_updates = self.config.ppo_epochs * self.config.ppo_num_minibatches
        self.minibatch_indices = config.device.indices(config.ppo_minibatch_size)

    def _update_policy(self, batch: RolloutBatch) -> Sequence[float]:
        pio, qo, beta = self.net(batch.states)
        cond_pio = pio[self.minibatch_indices, batch.options]
        cond_pio.set_action(batch.actions)
        # - (exp(logπo - old-logπo) x Adv)
        policy_loss = self._proximal_policy_loss(
            cond_pio, batch.advantages, batch.old_log_probs,
        )
        cond_qo = qo[self.minibatch_indices, batch.options]
        value_loss = (cond_qo - batch.returns).pow_(2).mul_(0.5).mean()
        cond_beta = beta[self.minibatch_indices, batch.prev_options]
        clip = -self.config.beta_logit_clip, self.config.beta_logit_clip
        clipped_logits = (cond_beta.dist.logits - batch.old_beta_logits).clamp_(*clip)
        # βo x clip(logit(βo) - logit(oldβo)) x Adv
        beta_loss = clipped_logits.mul_(batch.old_betas * batch.beta_advs).mean()
        # H(πo)
        pi_entropy = cond_pio.entropy().mean()
        # H(πμ)
        pimu_entropy = self._pimu_entropy(pio, batch.muo_xses)
        total_loss = (
            policy_loss
            - self.config.beta_loss_weight * beta_loss
            + self.config.value_loss_weight * value_loss
            - self.config.entropy_weight * pi_entropy
            - self.config.pimu_entropy_weight * pimu_entropy
            - self.config.beta_entropy_weight * cond_beta.dist.entropy().mean()
        )
        self._backward(total_loss, self.optimizer, self.net.parameters())
        return (
            policy_loss.item(),
            value_loss.item(),
            beta_loss.item(),
            pi_entropy.item(),
            pimu_entropy.item(),
        )

    def train(self, _last_states: Array[State]) -> None:
        # Setup batches
        prev_options, options = self.storage.batch_options()  # N x W
        x_all = self.storage.batch_states_all(self.penv)  # N x W + 1, |S|
        x, x_last = torch.split(x_all, [self.config.batch_size, self.nworkers])
        xs = self.storage._prepare_xs(self.tensor(self.xs_holder.xs), x)
        opt = self.storage._prepare_options(
            self.tensor(self.xs_holder.options, dtype=torch.long)
        )
        xf, beta_masks = self.storage._prepare_xf(x_last.squeeze_(1), x)
        # Get estimated values of V(x_N), β(x_N), and μ(・|x_N)
        with torch.no_grad():
            _, qo, beta, muo_xs = self.net.pqbm(x_all)
        vo = torch.einsum("bo,bo->b", qo, muo_xs.probs)
        # Compute advantage
        self._advantage(qo[-self.nworkers :], vo, beta.dist.probs[-self.nworkers :])
        # For not terminated options, mask advantage.
        beta_adv = self._beta_adv(xs, x, xf, opt).masked_fill_(beta_masks, 0.0)
        sampler = RolloutSampler(
            self.storage,
            self.penv,
            self.config,
            beta_adv,
            muo_xs.probs[: -self.nworkers],
            options,
            prev_options,
            self.batch_indices,
        )

        loss = np.zeros(5)
        for _ in range(self.config.ppo_epochs):
            for batch in sampler:
                loss += np.array(self._update_policy(batch))

        loss /= self.num_updates
        opt_model_log = self._train_opt_model()
        self.network_log(
            policy_loss=loss[0],
            value_loss=loss[1],
            beta_loss=loss[2],
            entropy=loss[3],
            pimu_entropy=loss[4],
            beta_min=sampler.old_betas.min().item(),
            beta_max=sampler.old_betas.max().item(),
            **opt_model_log,
        )
        self._submit_option_log()
        self.storage.reset()
