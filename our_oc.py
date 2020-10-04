from typing import Sequence

import numpy as np
import torch

# Local modules
import a2imoc
import ppimoc
from rainy.prelude import Array, State


class OurAOCAgent(a2imoc.A2ImocAgent):
    def train(self, _last_states: Array[State]) -> None:
        # Setup batches
        prev_options, options = self.storage.batch_options()  # N x W
        x_all = self.storage.batch_states_all(self.penv)  # N x W + 1, |S|
        x, _ = torch.split(x_all, [self.nsteps * self.nworkers, self.nworkers])
        # Get estimated values of V(x_N), β(x_N), and μ(・|x_N)
        with torch.no_grad():
            _, qo, beta, muo_xs = self.net.pqbm(x_all)
        vo = torch.einsum("bo,bo->b", qo, muo_xs.probs)
        # Compute advantage
        self._advantage(qo[-self.nworkers :], vo, beta.dist.probs[-self.nworkers :])
        policy_adv = self.storage.advs[:-1].flatten()
        # Compute loss
        pio, qo, beta = self.net(x)
        cond_qo = qo[self.batch_indices, options]
        cond_beta = beta[self.batch_indices, prev_options]
        masks = self.storage.batch_masks()
        delib_cost = self.storage.batch_opt_terminals() * self.config.opt_delib_cost
        beta_adv = cond_qo.detach() - vo[: -self.nworkers] + delib_cost
        # βo(Q - V)
        beta_loss = cond_beta.dist.probs.mul(masks * beta_adv).mean()
        cond_pio = pio[self.batch_indices, options]
        actions = self.storage.batch_actions()
        # -logπA
        policy_loss = -(cond_pio.log_prob(actions) * policy_adv).mean()
        ret = self.storage.returns[:-1].flatten()
        # (Q - ∑γR)^2
        value_loss = (cond_qo - ret).pow(2).mul(0.5).mean()
        muo_xs = muo_xs.probs[: -self.nworkers]
        # H(πo)
        pi_entropy = cond_pio.entropy().mean()
        # H(πμ)
        pimu_entropy = self._pimu_entropy(pio, muo_xs)
        ac_loss = (
            policy_loss
            + beta_loss
            + self.config.value_loss_weight * value_loss
            - self.config.entropy_weight * pi_entropy
            - self.config.pimu_entropy_weight * pimu_entropy
            - self.config.beta_entropy_weight * cond_beta.dist.entropy().mean()
        )
        self._backward(ac_loss, self.optimizer, self.net.parameters())
        opt_model_log = self._train_opt_model()
        beta_probs = beta.dist.probs.detach()
        self.network_log(
            policy_loss=policy_loss.item(),
            qo=qo.detach_().mean().item(),
            value_loss=value_loss.item(),
            beta_loss=beta_loss.item(),
            entropy=pi_entropy.item(),
            pimu_entropy=pimu_entropy.item(),
            beta_min=beta_probs.min().item(),
            beta_max=beta_probs.max().item(),
            **opt_model_log,
        )
        self._submit_option_log()
        self.storage.reset()


class OurPPOCAgent(ppimoc.PPImocAgent):
    def _update_policy(self, batch: ppimoc.RolloutBatch) -> Sequence[float]:
        pio, qo, beta = self.net(batch.states)
        cond_pio = pio[self.minibatch_indices, batch.options]
        cond_pio.set_action(batch.actions)
        # - (exp(logπo - old-logπo) x Adv)
        policy_loss = self._proximal_policy_loss(
            cond_pio, batch.advantages, batch.old_log_probs,
        )
        cond_qo = qo[self.minibatch_indices, batch.options]
        # (Q - ∑γR)^2
        value_loss = (cond_qo - batch.returns).pow_(2).mul_(0.5).mean()
        cond_beta = beta[self.minibatch_indices, batch.prev_options]
        # βo(Q - V)
        beta_loss = cond_beta.dist.probs.mul(batch.beta_advs).mean()
        # H(πo)
        pi_entropy = cond_pio.entropy().mean()
        # H(πμ)
        pimu_entropy = self._pimu_entropy(pio, batch.muo_xses)
        total_loss = (
            policy_loss
            + beta_loss
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
        x, _ = torch.split(x_all, [self.config.batch_size, self.nworkers])
        # Get estimated values of Q(x_N), β(x_N), and μ(・|x_N)
        with torch.no_grad():
            _, qo, beta, muo_xs = self.net.pqbm(x_all)
        vo = torch.einsum("bo,bo->b", qo, muo_xs.probs)
        # Compute advantage
        qo, qo_last = torch.split(qo, [self.nsteps * self.nworkers, self.nworkers])
        self._advantage(qo[-self.nworkers :], vo, beta.dist.probs[-self.nworkers :])
        delib_cost = self.storage.batch_opt_terminals() * self.config.opt_delib_cost
        beta_adv = qo[self.batch_indices, options] - vo[: -self.nworkers] + delib_cost
        sampler = ppimoc.RolloutSampler(
            self.storage,
            self.penv,
            self.config,
            beta_adv * self.storage.batch_masks(),
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
