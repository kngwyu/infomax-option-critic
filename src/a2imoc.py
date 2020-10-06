""" Implementation of A2IMOC.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from torch import LongTensor, Tensor
from torch.distributions import Categorical
from torch.nn import functional as F

# Local modules
from batch_holder import XsHolder, XsXfHolder
from option_select_impl import OptionSelectImpl
from rainy.agents import A2CLikeAgent, Netout
from rainy.config import Config
from rainy.net.policy import Policy
from rainy.prelude import Action, Array, State
from rollout import RolloutStorage


class A2ImocAgent(A2CLikeAgent[State], OptionSelectImpl):
    SAVED_MEMBERS = "net", "optimizer"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # Ensures that config object has specific values
        config.ensure(
            "option_selector", "epsg-logp", allowed=["epsg", "logp", "epsg-logp"],
        )
        config.ensure("adv_type", "upgoing", allowed=["upgoing", "vo", "ret"])
        config.ensure("upgoing_adv", True, allowed=[False, True])
        config.ensure("logmu_weight", 0.5)
        config.ensure("logmu_weight_min", None)
        config.ensure("beta_entropy_weight", 0.01)
        config.ensure("beta_loss_weight", 1.0)
        config.ensure("beta_logit_clip", 0.2)
        config.ensure("pimu_entropy_weight", 0.01)
        config.ensure("pimu_entropy_weight_min", 0.01)
        config.ensure("pimu_mc_rollout", 10)
        config.ensure("option_log_freq", 100)
        config.ensure("opt_model_capacity", 1024)
        config.ensure("opt_model_batch_size", 128)
        # Networks and optimizers
        self.net = config.net("option-critic")
        self.optimizer = config.optimizer(self.net.parameters())
        # Copy some configs to self to reduce LOC
        self.noptions = self.net.num_options
        self.nworkers = config.nworkers
        self.nsteps = config.nsteps
        # Some requirements for OptionSelectimpl
        self._opt_logp_cooler = config._get_cooler(
            config.logmu_weight, config.logmu_weight_min, self.nworkers,
        )
        # Only used when option_selector is "epsg" or "epsg-logp"
        self.opt_explorer = config.explorer()
        self.eval_opt_explorer = config.explorer(key="eval")
        self.worker_indices = config.device.indices(config.nworkers)
        self.batch_indices = config.device.indices(config.batch_size)
        self.storage = RolloutStorage(
            config.nsteps, config.nworkers, config.device, self.noptions
        )
        self.eval_prev_options = config.device.zeros(config.nworkers, dtype=torch.long)
        self.eval_is_initial_states = self.config.device.ones(
            self.nworkers, dtype=torch.bool,
        )
        self.option_counter = np.zeros(self.noptions, dtype=np.int64)
        config.logger.summary_setting(
            "option",
            ["update_steps"],
            interval=config.option_log_freq,
            color="cyan",
            describe_range=True,
        )
        self.xs_holder = None
        self.xs_xf_holder = XsXfHolder(
            num_options=self.noptions,
            state_shape=config.state_dim,
            batch_size=config.opt_model_batch_size,
            capacity=config.opt_model_capacity,
        )
        self.state_cache = None
        self.opt_target = self.tensor(self.xs_xf_holder.opt_target(), dtype=torch.long)

        policy_cls = self.net.policy_dist.__class__.__name__

        if policy_cls == "CategoricalDist":

            def _pimu_entropy(pio: Policy, muo_xs: Tensor) -> Tensor:
                """ Returns exact H(πμ)
                """
                pimu_probs = torch.einsum("boa,bo->ba", pio.dist.probs, muo_xs)
                return Categorical(probs=pimu_probs).entropy().mean()

        elif "Gaussian" in policy_cls:

            def _pimu_entropy(pio: Policy, muo_xs: Tensor) -> Tensor:
                """ Returns monte-carlo approximation of H(πμ)
                """
                neg_logpimu = []
                mu = Categorical(probs=muo_xs)
                for _ in range(self.config.pimu_mc_rollout):
                    options = mu.sample()
                    actions = pio.dist.sample()[self.minibatch_indices, options]
                    # Actions and muo_xs must be unsqueezed with the option dim
                    log_pia = pio.dist.log_prob(actions.unsqueeze(1))  # B x O x A
                    log_pia_po = log_pia + torch.log(muo_xs).unsqueeze(-1)  # B x O x A
                    # -log∑exp(log(μ(o|x)π(a|o, x)) = -log(πμ(a|x))
                    neg_logpimu.append(-torch.logsumexp(log_pia_po, 1))
                return torch.stack(neg_logpimu).mean()

        else:
            raise NotImplementedError(f"{policy_cls} is not supported")

        self._pimu_entropy = _pimu_entropy

    def logmu_weight(self) -> float:
        return self._opt_logp_cooler()

    def eval_reset(self) -> None:
        self.eval_prev_options.fill_(0)
        self.eval_is_initial_states.fill_(True)

    def _reset(self, initial_states: Array[State]) -> None:
        """ Ensure all storages are initialized before training.
            For adaptation tasks.
        """
        self.storage.initialize()
        self.storage.set_initial_state(initial_states)
        self.xs_xf_holder.clear()
        self.xs_holder = XsHolder(
            self.penv.extract(initial_states),
            self.storage.options[0].cpu().numpy(),
            self.xs_xf_holder.append,
        )

    def _update_option_count(self, new_options: Tensor) -> None:
        new_options = new_options.cpu().numpy()
        for opt in new_options:
            self.option_counter[opt] += 1

    @torch.no_grad()
    def _eval_policy(self, states: Array) -> Tuple[Policy, Tensor]:
        batch_size = states.shape[0]
        pio, qo, beta, _ = self.net.pqbm(states)
        options, _ = self._eval_sample_options(qo, beta)
        self.eval_prev_options[:batch_size] = options
        return pio, options

    def eval_action(self, state: Array, net_outputs: Optional[Netout] = None) -> Action:
        if state.ndim == len(self.net.state_dim):
            # treat as batch_size == 1
            state = np.stack([state])
        pio, options = self._eval_policy(state)
        if net_outputs is not None:
            net_outputs["options"] = self.eval_prev_options
        cond_pio = pio[0, options[0]]
        return cond_pio.eval_action(self.config.eval_deterministic)

    def eval_action_parallel(self, states: Array) -> Array[Action]:
        batch_size = states.shape[0]
        pio, options = self._eval_policy(states)
        cond_pio = pio[self.config.device.indices(batch_size), options]
        return cond_pio.eval_action(self.config.eval_deterministic)

    @property
    def prev_options(self) -> LongTensor:
        return self.storage.options[-1]

    @torch.no_grad()
    def actions(self, states: Array[State]) -> Tuple[Array[Action], dict]:
        """
        1. Sample termination from βo
        2. Sample the next option o' from μo
        3. Sample the next action from πo'
        """
        x = self.penv.extract(states)
        pio, qo, beta, po_xs = self.net.pqbm(x)
        options, opt_terminals = self._sample_options(qo, beta, po_xs)
        self._update_option_count(options[opt_terminals])
        prev_terminals = (1.0 - self.storage.masks[-1]).bool().cpu().numpy()
        self.xs_holder.update(
            x, prev_terminals, options.cpu().numpy(), opt_terminals.cpu().numpy(),
        )
        cond_pio = pio[self.worker_indices, options]
        actions = cond_pio.action().squeeze().cpu().numpy()
        net_outputs = dict(
            policy=cond_pio,
            value=qo,
            options=options,
            opt_terminals=opt_terminals,
            opt_policy=pio,
            beta=beta,
            po_xs=po_xs,
        )
        return actions, net_outputs

    def _one_step(self, states: Array[State]) -> Array[State]:
        actions, net_outputs = self.actions(states)
        transition = self.penv.step(actions).map_r(lambda r: r * self.reward_scale)
        self.storage.push(*transition[:3], **net_outputs)
        self.returns += transition.rewards
        self.episode_length += 1
        self._report_reward(transition.terminals, transition.infos)
        return transition.states

    @torch.no_grad()
    def _next_uo(
        self, next_qo: Tensor, beta: Tensor, next_vo: Optional[Tensor] = None
    ) -> Tensor:
        next_qo_ = next_qo[self.worker_indices, self.prev_options]
        beta = beta[self.worker_indices, self.prev_options]
        if next_vo is not None:
            next_vo = next_vo.flatten()
        else:
            next_vo = qo.mean(-1)
        # Uo(St+N) = (1.0 - β) Qo(St+N, O) + β Vo(St+N)
        return (1.0 - beta).mul_(next_qo_).add_(beta * next_vo)

    def _beta_adv(self, xs: Tensor, x: Tensor, xf: Tensor, opt: Tensor) -> Tensor:
        with torch.no_grad():
            log_po_xs_x = self.net.po_xsxf(xs, x).log_prob(opt)
            log_po_xsxf = self.net.po_xsxf(xs, xf).log_prob(opt)
        return log_po_xs_x - log_po_xsxf

    def _advantage(
        self, next_qo: Tensor, vo: Tensor, next_beta: Tensor,
    ) -> Optional[Tensor]:
        gamma, lambda_ = self.config.discount_factor, self.config.gae_lambda
        next_vo = vo[-self.nworkers :]
        if self.config.use_gae:
            next_uo = self._next_uo(next_qo, next_beta, next_vo)
            self.storage.calc_gae_vo(next_uo, vo, gamma, lambda_, self.config.adv_type)
        else:
            next_uo = self._next_uo(next_qo, next_beta, next_vo)
            self.storage.calc_ret_vo(next_uo, vo, gamma, self.config.adv_type)

    def _train_opt_model(self) -> Dict[str, float]:
        if not self.xs_xf_holder.ready():
            return dict(
                muo_xs_loss=0.0, po_xsxf_loss=0.0, po_xsxf_min=1.0, po_xsxf_max=0.0,
            )
        xs, xf = self.xs_xf_holder.get_batch()
        muo_xs, po_xs_xf = self.net.muo_xs_po_xsxf(self.tensor(xs), self.tensor(xf))
        mu_loss = F.cross_entropy(muo_xs.logits, self.opt_target)
        p_loss = F.cross_entropy(po_xs_xf.logits, self.opt_target)
        self._backward(mu_loss + p_loss, self.optimizer, self.net.parameters())
        probs = po_xs_xf.probs.detach()
        return dict(
            muo_xs_loss=mu_loss.item(),
            po_xsxf_loss=p_loss.item(),
            po_xsxf_min=probs.min().item(),
            po_xsxf_max=probs.max().item(),
        )

    def _submit_option_log(self) -> None:
        counter = dict(
            map(lambda i: (f"option-{i}", self.option_counter[i]), range(self.noptions))
        )
        self.logger.submit("option", **counter, update_steps=self.update_steps)

    def train(self, last_states: Array[State]) -> None:
        # Setup batches
        prev_options, options = self.storage.batch_options()  # N x W
        x_all = self.storage.batch_states_all(self.penv)  # N x W + 1, |S|
        x, x_last = torch.split(x_all, [self.nsteps * self.nworkers, self.nworkers])
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
        policy_adv = self.storage.advs[:-1].flatten()
        # Compute loss
        pio, qo, beta = self.net(x)
        cond_beta = beta[self.batch_indices, prev_options]
        beta_probs = cond_beta.dist.probs.detach()
        # βo x logit(βo) x Adv
        beta_loss = (beta_probs * cond_beta.dist.logits).mul_(beta_adv).mean()
        cond_pio = pio[self.batch_indices, options]
        actions = self.storage.batch_actions()
        # - (logπo x Adv)
        policy_loss = -(cond_pio.log_prob(actions) * policy_adv).mean()
        cond_qo = qo[self.batch_indices, options]
        ret = self.storage.returns[:-1].flatten()
        # (Qo - ∑γR)^2
        value_loss = (cond_qo - ret).pow(2).mul(0.5).mean()
        muo_xs = muo_xs.probs[: -self.nworkers]
        # H(πo)
        pi_entropy = cond_pio.entropy().mean()
        # H(πμ)
        pimu_entropy = self._pimu_entropy(pio, muo_xs)
        ac_loss = (
            policy_loss
            - self.config.beta_loss_weight * beta_loss
            + self.config.value_loss_weight * value_loss
            - self.config.entropy_weight * pi_entropy
            - self.config.pimu_entropy_weight * pimu_entropy
            - self.config.beta_entropy_weight * cond_beta.dist.entropy().mean()
        )
        self.config.pimu_entropy_weight = max(
            0.0, self.config.pimu_entropy_weight - 0.0001
        )
        self._backward(ac_loss, self.optimizer, self.net.parameters())
        opt_model_log = self._train_opt_model()
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

    def loadmember_hook(self, member_str: str, saved_item: dict) -> None:
        """ For loading outdated save files. Nothing to see.
        """
        appended, del_keys = {}, []
        for key in saved_item.keys():
            members = key.split(".")
            if members[0] == "value_head":
                new_name = ".".join(["qo_head"] + members[1:])
                appended[new_name] = saved_item[key]
                del_keys.append(key)
            elif members[0] == "vmu_head":
                del_keys.append(key)
        saved_item.update(appended)
        for del_key in del_keys:
            del saved_item[del_key]
