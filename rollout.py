from typing import Tuple

import torch

from torch import LongTensor, Tensor

from rainy.lib.rollout import RolloutStorage as A2CRollout
from rainy.net.policy import BernoulliPolicy
from rainy.prelude import State
from rainy.utils import Device


class RolloutStorage(A2CRollout[State]):
    def __init__(
        self, nsteps: int, nworkers: int, device: Device, num_options: int,
    ) -> None:
        super().__init__(nsteps, nworkers, device)
        self.options = [self.device.zeros(nworkers, dtype=torch.long)]
        self.opt_terminals = [self.device.ones(nworkers, dtype=torch.bool)]
        self.noptions = num_options
        self.worker_indices = self.device.indices(nworkers)
        self._term_advs = torch.zeros_like(self.advs)
        self.betas = []
        self.beta_logits = []

    def reset(self) -> None:
        super().reset()
        self.options = [self.options[-1]]
        self.opt_terminals = [self.opt_terminals[-1]]
        self.betas.clear()
        self.beta_logits.clear()

    def initialize(self) -> None:
        super().initialize()
        self.options = [self.device.zeros(self.nworkers, dtype=torch.long)]
        self.opt_terminals = [self.device.ones(self.nworkers, dtype=torch.bool)]
        self.betas.clear()
        self.beta_logits.clear()

    def push(
        self,
        *args,
        options: LongTensor,
        opt_terminals: Tensor,
        beta: BernoulliPolicy,
        **kwargs,
    ) -> None:
        super().push(*args, **kwargs)
        self.options.append(options)
        self.opt_terminals.append(opt_terminals)
        self.betas.append(beta.dist.probs.detach())
        self.beta_logits.append(beta.dist.logits.detach())

    def batch_options(self) -> Tuple[Tensor, Tensor]:
        batched = torch.cat(self.options)
        return batched[: -self.nworkers], batched[self.nworkers :]

    def batch_opt_terminals(self) -> Tensor:
        return torch.cat(self.opt_terminals[1:])

    def calc_ret_vo(
        self, next_uo: Tensor, vo: Tensor, gamma: float, adv_type: str = "upgoing",
    ) -> Tensor:
        vo = vo.view(self.nsteps + 1, self.nworkers)
        self.returns[-1] = next_uo
        rewards = self.device.tensor(self.rewards)
        opt_terminals = self.device.zeros((self.nworkers,), dtype=torch.bool)
        for i in reversed(range(self.nsteps)):
            opt = self.options[i + 1]
            if adv_type == "upgoing":
                v_changed = torch.max(vo[i + 1], self.returns[i + 1])
            elif adv_type == "vo":
                v_changed = vo[i + 1]
            elif adv_type == "ret":
                v_changed = self.returns[i + 1]
            else:
                raise NotImplementedError()
            ret_i1 = torch.where(opt_terminals, v_changed, self.returns[i + 1],)
            opt_terminals = self.opt_terminals[i + 1]
            self.returns[i] = gamma * self.masks[i + 1] * ret_i1 + rewards[i]
            vi = self.values[i][self.worker_indices, opt]
            self.advs[i] = self.returns[i] - vi

    def termination_adv(self, gamma: float, threshold: float = 0.5) -> Tensor:
        self._term_advs.fill_(0.0)
        for i in reversed(range(self.nsteps)):
            beta = self.betas[i][self.worker_indices, self.options[i]]
            bonus = beta.clamp(min=threshold) - threshold
            self._term_advs[i] = torch.where(
                self.opt_terminals[i + 1], bonus, gamma * self._term_advs[i + 1],
            )
        return self._term_advs[:-1].flatten()

    def calc_gae_vo(
        self,
        next_uo: Tensor,
        vo: Tensor,
        gamma: float,
        lambda_: float,
        adv_type: str = "upgoing",
    ) -> None:
        rewards = self.device.tensor(self.rewards)
        self.advs.fill_(0.0)
        vo_advs = torch.zeros_like(self.advs)
        vo = vo.view(self.nsteps + 1, self.nworkers)
        vi1 = next_uo
        opt_terminals = self.device.zeros((self.nworkers,), dtype=torch.bool)
        for i in reversed(range(self.nsteps)):
            opt, opt_q = self.options[i + 1], self.values[i]
            vi = opt_q[self.worker_indices, opt]
            vi1_ = torch.where(opt_terminals, vo[i + 1], vi1)
            if adv_type == "upgoing":
                adv_changed = vo_advs[i + 1].clamp(min=0.0)
            elif adv_type == "vo":
                adv_changed = vo_advs[i + 1] * 0.0
            elif adv_type == "ret":
                adv_changed = vo_advs[i + 1]
            else:
                raise NotImplementedError()
            adv_i1 = torch.where(opt_terminals, adv_changed, self.advs[i + 1])
            gamma_i1 = gamma * self.masks[i + 1]
            td_error = rewards[i] + gamma_i1 * vi1_ - vi
            self.advs[i] = td_error + gamma_i1 * lambda_ * adv_i1
            self.returns[i] = self.advs[i] + vi
            vo_tde = rewards[i] + gamma_i1 * vo[i + 1] - vo[i]
            vo_advs[i] = vo_tde + gamma_i1 * lambda_ * vo_advs[i + 1]
            vi1 = vi
            opt_terminals = self.opt_terminals[i + 1]

    def _prepare_xf(self, xf: Tensor, batch_states: Tensor) -> Tuple[Tensor, Tensor]:
        state_shape = batch_states.shape[1:]
        states = batch_states.view(self.nsteps, self.nworkers, -1)
        masks = self.device.zeros((self.nsteps + 1, self.nworkers), dtype=torch.bool)
        xf_last = xf.view(self.nworkers, -1)
        res = []
        for i in reversed(range(self.nsteps)):
            opt_terminals = self.opt_terminals[i + 1]
            xf_last = torch.where(opt_terminals.unsqueeze(1), states[i], xf_last)
            masks[i] = masks[i + 1] | opt_terminals
            res.append(xf_last)
        res.reverse()
        # If mask[i][j] == True, option[i][j] is not terminated yet
        masks = masks[:-1].flatten().logical_not_()
        return torch.cat(res).view(self.nsteps * self.nworkers, *state_shape), masks

    def _prepare_xs(self, xs: Tensor, batch_states: Tensor) -> Tensor:
        state_shape = batch_states.shape[1:]
        states = batch_states.view(self.nsteps, self.nworkers, -1)
        xs_last = xs.view(self.nworkers, -1)
        res = []
        for i in range(self.nsteps):
            opt_terminals = self.opt_terminals[i].unsqueeze(1)
            xs_last = torch.where(opt_terminals, states[i], xs_last)
            res.append(xs_last)
        return torch.cat(res).view(self.nsteps * self.nworkers, *state_shape)

    def _prepare_options(self, opt: LongTensor) -> Tensor:
        res = []
        for i in range(self.nsteps):
            opt_terminals = self.opt_terminals[i]
            opt = torch.where(opt_terminals, self.options[i], opt)
            res.append(opt)
        return torch.cat(res)
