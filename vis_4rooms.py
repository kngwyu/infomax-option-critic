from typing import Optional, Tuple

import numpy as np
import torch

from torch import Tensor

import rainy

from rainy.lib.hooks import EvalHook
from rainy.prelude import State
from rlpy.gym import RLPyEnv
from rlpy.tools import with_pdf_fonts
from vis_utils import get_n_row_cols


def _to_np(batch_size: int, env: RLPyEnv) -> callable:
    ngoals = env.domain.num_goals

    if ngoals > 0:

        def to_np(tensor):
            shape = batch_size // ngoals, ngoals, *tensor.shape[1:]
            return tensor.view(shape).mean(1).cpu().numpy()

    else:

        def to_np(tensor):
            return tensor.cpu().numpy()

    return to_np


class _AllStates(EvalHook):
    def setup(self, config: rainy.Config) -> None:
        self.device = config.device

    def _all_states(self, env: RLPyEnv, extract: callable) -> Tensor:
        xf = []
        for state in env.domain.all_states():
            xf.append(extract(state))
        return self.device.tensor(np.stack(xf))


def _show_beta_pi(env: RLPyEnv, beta: np.ndarray, pi: np.ndarray, **kwargs) -> None:
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    domain = env.domain
    for i in range(pi.shape[1]):
        key = domain.show_policy(
            pi[:, i, :],
            beta[:, i],
            index=i + 1,
            ticks=False,
            title=f"Option: {i}",
            scale=1.6,
            cmap="YlGnBu",
            cmap_vmin=0.0,
            cmap_vmax=1.0,
            colorbar=False,
            notext=True,
            figure_title="π & β",
            **kwargs,
        )
        if kwargs["ncols"] == i + 1:
            divider = make_axes_locatable(domain.policy_ax[key])
            cax = divider.append_axes("right", size="6%", pad=0.1)
            domain.policy_fig[key[0]].colorbar(
                domain.policy_img[key], cax=cax, orientation="vertical",
            )
            cax.set_ylabel("β", rotation=0, position=(1.0, 0.55), fontsize=20)


def _show_qo_pi(env: RLPyEnv, qo: np.ndarray, pi: np.ndarray, **kwargs) -> None:
    for i in range(pi.shape[1]):
        env.domain.show_policy(
            pi[:, i, :],
            qo[:, i],
            index=i + 1,
            ticks=False,
            title=f"Option: {i}",
            scale=1.6,
            cmap_vmin=-2.0,
            cmap_vmax=2.0,
            **kwargs,
        )


class OCVis(_AllStates):
    def __init__(
        self, num_options: int, is_flat_layout: bool = False, beta_pi: bool = False,
    ) -> None:
        self.num_options = num_options
        self.counter = 0
        self.nrowcols = get_n_row_cols(num_options, is_flat_layout=is_flat_layout)
        self.beta_pi = beta_pi

    def reset(
        self, agent: rainy.agents.Agent, env: rainy.envs.EnvExt, initial_state: State,
    ) -> None:
        x = self._all_states(env.unwrapped, env.extract)
        to_np = _to_np(x.size(0), env.unwrapped)

        with torch.no_grad():
            pi, qo, beta = agent.net(x)

        beta = to_np(beta.dist.probs)
        pi = to_np(pi.dist.probs)

        if self.beta_pi:
            _show_beta_pi(env.unwrapped, beta, pi, **self.nrowcols)
        else:
            for i in range(self.num_options):
                env.unwrapped.domain.show_heatmap(
                    beta[:, i],
                    "β(Xf)",
                    normalize_method="none",
                    colorbar=True,
                    cmap="YlGnBu",
                    **self.nrowcols,
                    index=i + 1,
                    ticks=False,
                    title=f"Option: {i}",
                    cmap_vmin=0.0,
                    legend=self.counter == 0 and i == 1,
                )
            qo = to_np(qo)
            _show_qo_pi(env.unwrapped, qo, pi, **self.nrowcols)

        self.counter += 1


class A2CVis(_AllStates):
    def __init__(self, save: bool = False) -> None:
        self.counter = 0
        self.save = save

    def reset(
        self, agent: rainy.agents.Agent, env: rainy.envs.EnvExt, initial_state: State
    ) -> None:
        x = self._all_states(env.unwrapped, env.extract)
        to_np = _to_np(x.size(0), env.unwrapped)

        with torch.no_grad():
            pi, v, *_ = agent.net(x)

        pi = to_np(pi.dist.probs)
        v = to_np(v)
        env.unwrapped.domain.show_policy(pi, v, ticks=False, title="Policy")

        self.counter += 1


class RndVis(A2CVis):
    def reset(
        self, agent: rainy.agents.Agent, env: rainy.envs.EnvExt, initial_state: State
    ) -> None:
        x = self._all_states(env.unwrapped, env.extract)
        to_np = _to_np(x.size(0), env.unwrapped)

        with torch.no_grad():
            pi, v, int_v, *_ = agent.net(x)

        pi = to_np(pi.dist.probs)
        v = to_np(v)

        if self.save:
            env.unwrapped.domain.show_policy(
                pi, None, ticks=False, cmap_vmin=-1.5, cmap_vmax=1.5, colorbar=True
            )
            with with_pdf_fonts():
                env.unwrapped.domain.policy_fig["Policy"].savefig(
                    f"policy{self.counter}.pdf"
                )
        else:
            env.unwrapped.domain.show_policy(
                pi, v, ticks=False, cmap_vmin=-1.5, cmap_vmax=1.5, colorbar=True
            )

        int_v = to_np(int_v)
        env.unwrapped.domain.show_heatmap(
            v, "Int V", cmap_vmin=-0.5, cmap_vmax=0.5,
        )

        intrew = agent.irew_gen.eval_gen_rewards(x)

        intrew = to_np(intrew)
        env.unwrapped.domain.show_heatmap(
            intrew,
            "Intrinsic Rewards",
            cmap_vmin=-0.5,
            cmap_vmax=0.5,
            ticks=False,
            legend=False,
            colorbar=True,
            normalize_method="none",
        )
        if self.save:
            with with_pdf_fonts():
                env.unwrapped.domain.heatmap_fig["Intrinsic Rewards"].savefig(
                    f"ir{self.counter}.pdf"
                )

        self.counter += 1


class _XsXf(EvalHook):
    def setup(self, config: rainy.Config) -> None:
        self.device = config.device

    def _xs_xf(
        self, env: RLPyEnv, initial_state: State, extract: callable,
    ) -> Tuple[Tensor, Tensor]:
        xf = []
        for state in env.domain.all_states():
            xf.append(extract(state))
        xs = np.stack([extract(initial_state) for _ in range(len(xf))])
        return self.device.tensor(xs), self.device.tensor(np.stack(xf))


class ImocVis(_XsXf):
    def __init__(
        self,
        num_options: int,
        save: bool = False,
        initial_state: Optional[State] = None,
        is_flat_layout: bool = False,
        beta_pi: bool = False,
    ) -> None:
        self.num_options = num_options
        self.counter = 0
        self.xs = initial_state
        self.save = save
        self.nrowcols = get_n_row_cols(num_options, is_flat_layout=is_flat_layout)
        self.beta_pi = beta_pi

    def _vis_p(self, env, agent, xs, xf, to_np) -> Tensor:
        with torch.no_grad():
            muo_xs, po_xsxf = agent.net.muo_xf_po_xsxf(xs, xf)
            po_xsxf = po_xsxf.probs.cpu().numpy()
            muo_xs = muo_xs.probs.cpu().numpy()
        for i in range(self.num_options):
            env.unwrapped.domain.show_heatmap(
                po_xsxf[:, i],
                "P(o|Xf, Xs)(Xs=(0,0))",
                normalize_method="none",
                colorbar=True,
                cmap="YlGnBu",
                title=f"Option: {i}",
                **self.nrowcols,
                index=i + 1,
                ticks=False,
                cmap_vmin=0.0,
                legend=self.counter == 0 and i == 1,
            )

            env.unwrapped.domain.show_heatmap(
                muo_xs[:, i],
                "P(o|Xs)",
                normalize_method="none",
                colorbar=True,
                cmap="YlGnBu",
                title=f"Option: {i}",
                **self.nrowcols,
                index=i + 1,
                ticks=False,
                cmap_vmin=0.0,
                legend=self.counter == 0 and i == 1,
            )

    def _vis_beta(self, beta, domain):
        for i in range(self.num_options):
            domain.show_heatmap(
                beta[:, i],
                "β(Xf)",
                normalize_method="none",
                colorbar=i + 1 == self.nrowcols["ncols"],
                legend=False,
                notext=True,
                cmap="YlGnBu",
                **self.nrowcols,
                index=i + 1,
                ticks=False,
                title=f"Option: {i}",
                cmap_vmin=0.0,
                scale=1.1,
            )

    def reset(
        self, agent: rainy.agents.Agent, env: rainy.envs.EnvExt, initial_state: State
    ) -> None:
        # env.unwrapped.domain.show_domain()
        # env.unwrapped.domain.domain_fig.savefig("FourRoomsGridWorld.pdf")
        initial_state = initial_state if self.xs is None else self.xs
        xs, xf = self._xs_xf(env.unwrapped, initial_state, env.extract)
        to_np = _to_np(xs.size(0), env.unwrapped)
        domain = env.unwrapped.domain

        with torch.no_grad():
            pi, qo, beta, muo_xs = agent.net.pqbm(xf)

        beta = to_np(beta.dist.probs)
        pi = to_np(pi.dist.probs)

        if self.beta_pi:
            _show_beta_pi(env.unwrapped, beta, pi, **self.nrowcols)
        else:
            self._vis_beta(beta, domain)
            qo = to_np(qo)
            _show_qo_pi(env.unwrapped, qo, pi, **self.nrowcols)

        self._vis_p(env, agent, xs, xf, to_np)

        self.counter += 1


class ImocRndVis(ImocVis):
    def reset(
        self, agent: rainy.agents.Agent, env: rainy.envs.EnvExt, initial_state: State
    ) -> None:
        initial_state = initial_state if self.xs is None else self.xs
        xs, xf = self._xs_xf(env.unwrapped, initial_state, env.extract)
        to_np = _to_np(xs.size(0), env.unwrapped)

        with torch.no_grad():
            pi, qo, int_qo, beta = agent.net(xf)

        beta = to_np(beta.dist.probs)
        for i in range(self.num_options):
            env.unwrapped.domain.show_heatmap(
                beta[:, i],
                "β(Xf)",
                normalize_method="none",
                colorbar=True,
                cmap="YlGnBu",
                **self.nrowcols,
                index=i + 1,
                ticks=False,
                title=f"Option: {i}",
                cmap_vmin=0.0,
                legend=False,
            )

        pi = to_np(pi.dist.probs)
        qo = to_np(qo)
        for i in range(self.num_options):
            env.unwrapped.domain.show_policy(
                pi[:, i, :],
                qo[:, i],
                **self.nrowcols,
                index=i + 1,
                scale=1.5,
                ticks=False,
                cmap_vmin=-1.5,
                cmap_vmax=1.5,
                title=f"Option: {i}",
            )

        int_qo = to_np(int_qo)
        for i in range(self.num_options):
            env.unwrapped.domain.show_heatmap(
                int_qo[:, i],
                "Intrinsic V",
                **self.nrowcols,
                index=i + 1,
                scale=1.5,
                ticks=False,
                cmap_vmin=-1.5,
                cmap_vmax=1.5,
                title=f"Option: {i}",
            )

        self._vis_p(env, agent, xs, xf, to_np)

        intrew = to_np(agent.irew_gen.eval_gen_rewards(xf))
        env.unwrapped.domain.show_heatmap(
            intrew,
            "Intrinsic Rewards",
            cmap_vmin=-0.5,
            cmap_vmax=0.5,
            normalize_method="none",
            ticks=False,
            legend=False,
            colorbar=True,
        )

        self.counter += 1


class ActcVis(_XsXf):
    def __init__(self, num_options: int, vis_p: bool = False,) -> None:
        self.num_options = num_options
        self.vis_p = vis_p
        self.initial = True

    def setup(self, config: rainy.Config) -> None:
        self.device = config.device

    def _vis_beta(self, beta, domain):
        for i in range(self.num_options):
            domain.show_heatmap(
                beta[:, i],
                "β(Xf)",
                normalize_method="none",
                colorbar=True,
                cmap="YlGnBu",
                nrows=2,
                ncols=2,
                index=i + 1,
                ticks=False,
                title=f"Option: {i}",
                cmap_vmin=0.0,
                legend=self.initial and i == 1,
            )

    def reset(
        self, agent: rainy.agents.Agent, env: rainy.envs.EnvExt, initial_state: State,
    ) -> None:
        xs, xf = self._xs_xf(env.unwrapped, initial_state, env.extract)
        to_np = _to_np(xf.size(0), env.unwrapped)

        with torch.no_grad():
            beta, p, _, _ = agent.tc_net(xs, xf)

        beta = to_np(beta.dist.probs)
        self._vis_beta(beta, env.unwrapped.domain)

        with torch.no_grad():
            pi, q = agent.ac_net(xf)
        pi = to_np(pi.dist.probs)
        q = to_np(q)
        for i in range(self.num_options):
            env.unwrapped.domain.show_policy(
                pi[:, i, :],
                q[:, i],
                nrows=2,
                ncols=2,
                index=i + 1,
                scale=1.6,
                ticks=False,
                title=f"Option: {i}",
            )

        if self.vis_p:
            p = to_np(p)
            for i in range(self.num_options):
                env.unwrapped.domain.show_heatmap(
                    p[:, i],
                    "P(Xs|Xf)(Xs=(0,0))",
                    normalize_method="uniform",
                    cmap="PuOr",
                    title=f"Option: {i}",
                    nrows=2,
                    ncols=2,
                    index=i + 1,
                    ticks=False,
                    legend=self.initial and i == 1,
                )

        self.initial = True
