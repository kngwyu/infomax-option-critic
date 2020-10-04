import itertools as it
from typing import List, Optional, Tuple

from gym.spaces import Box
import numpy as np
import torch

from mujoco_maze.maze_env import MazeEnv
import rainy

from vis_utils import get_n_row_cols, ValueHeatMap, Trajectory
from rainy.lib.hooks import EvalHook
from rainy.prelude import Self, State

N_SENSOR_SAMPLES_DEFAULT = 100
SEED = 123456


def _sample_sensor_obs(space: Box, n: int, seed: int = SEED) -> np.ndarray:
    space.seed(seed)
    return np.stack([space.sample() for _ in range(n)])


def _sample_states(
    env: MazeEnv, cell_discr: int, n_sensor_samples: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]], Tuple[int, int]]:
    structure, scaling = env.unwrapped._maze_structure, env.unwrapped._maze_size_scaling
    space = env.observation_space
    # Construct a dummy Box space to sample sensory observations
    sensor_obs = _sample_sensor_obs(
        Box(space.low[2:], space.high[2:]), n_sensor_samples,
    )
    init_x = env.unwrapped._init_torso_x
    init_y = env.unwrapped._init_torso_y
    unit = 1.0 / cell_discr
    offset = unit / 2 - 0.5
    res = []
    mapping = []
    h, w = len(structure), len(structure[0])
    coordinates = []
    for i, j in it.product(range(h), range(w)):
        if structure[i][j].is_block():
            continue
        arrs = []
        for dyi, dxi in it.product(range(cell_discr), range(cell_discr)):
            y, x = i + offset + dyi * unit, j + offset + dxi * unit
            xy = np.array([x * scaling - init_x, y * scaling - init_y])
            xy_repeated = np.tile(xy, (n_sensor_samples, 1))
            arr = np.concatenate((xy_repeated, sensor_obs), axis=1)
            arrs.append(arr)
            mapping.append((j * cell_discr + dxi, i * cell_discr + dyi))
            coordinates.append((x * cell_discr, y * cell_discr))
        res.append(np.stack(arrs))
    dat = HeatmapData(1, h, w, cell_discr, mapping)
    cds = np.array(coordinates) + 0.5
    return np.concatenate(res), dat, cds


class HeatmapData:
    def __init__(
        self, o: int, h: int, w: int, cell_discr: int, mapping: List[Tuple[int, int]],
    ) -> None:
        self.mapping = mapping
        self.data = np.zeros((o, h * cell_discr, w * cell_discr))
        self.data_shape = h * cell_discr, w * cell_discr
        self.noptions = o
        self.h_w_cell_discr = h, w, cell_discr

    def update(self, data: np.ndarray) -> np.ndarray:
        data = data.reshape(-1, self.noptions)  # Batch x Options
        for i, (x, y) in enumerate(self.mapping):
            self.data[:, y, x] = data[i]
        return self.data

    def copy(self, noptions: Optional[int] = None) -> Self:
        noptions = noptions or self.noptions
        return HeatmapData(noptions, *self.h_w_cell_discr, self.mapping)


class A2CVis(EvalHook):
    def __init__(
        self,
        cell_discr: int = 2,
        n_sensor_samples: int = N_SENSOR_SAMPLES_DEFAULT,
        vmin: float = -1.2,
        vmax: float = 1.2,
    ) -> None:
        self.counter = 0
        self.cell_discr = cell_discr
        self.n_sensor_samples = n_sensor_samples
        self.vmin = vmin
        self.vmax = vmax

    def setup(self, config: rainy.Config) -> None:
        self.device = config.device

    def reset(
        self, agent: rainy.agents.Agent, env: rainy.envs.EnvExt, initial_state: State
    ) -> None:
        if self.counter == 0:
            states, self.heatmap_data, _ = _sample_states(
                env.unwrapped, self.cell_discr, self.n_sensor_samples,
            )
            self.states = self.device.tensor(states)  # cells, sensor_samples, |s|
            self.value_heatmap = ValueHeatMap(
                self.heatmap_data.data_shape,
                name="Value",
                vmin=self.vmin,
                vmax=self.vmax,
                reverse_y=True,
            )

        with torch.no_grad():
            pi, v, *_ = agent.net(self.states.view(-1, self.states.size(-1)))

        v = v.view(self.states.shape[:2]).mean(dim=1).cpu().numpy()
        self.value_heatmap.update(self.heatmap_data.update(v))

        self.counter += 1


class ImocVis(EvalHook):
    POLICY_SCALE: float = 1.5

    def __init__(
        self,
        noptions: int = 4,
        cell_discr: int = 2,
        n_sensor_samples: int = N_SENSOR_SAMPLES_DEFAULT,
        vmin: float = -0.8,
        vmax: float = 0.8,
        gain_index: int = 0,
        rot_index: int = 1,
        vis_policy: bool = False,
        vis_trajectory: bool = True,
        vis_value: bool = True,
        is_flat_layout: bool = True,
    ) -> None:
        self.counter = 0
        self.noptions = noptions
        self.cell_discr = cell_discr
        self.n_sensor_samples = n_sensor_samples
        self.vmin = vmin
        self.vmax = vmax
        self._gain_index = gain_index
        self._rot_index = rot_index
        self._ori_index = 0
        self._vis_value = vis_value
        self._vis_policy = vis_policy
        self._vis_traj = vis_trajectory
        self._is_flat = is_flat_layout
        self.nrowcols = get_n_row_cols(self.noptions, is_flat_layout=is_flat_layout)

    def setup(self, config: rainy.Config) -> None:
        self.device = config.device

    def _initialize(self, env: MazeEnv) -> None:
        if self._vis_policy:
            self._ori_index = env.wrapped_env.ORI_IND
        states, heatmap_data, self._xy = _sample_states(
            env, self.cell_discr, self.n_sensor_samples,
        )
        data_shape = heatmap_data.data_shape
        self.states = self.device.tensor(states)  # cells, sensor_samples, |s|
        self.qo_heatmap_data = heatmap_data.copy(self.noptions)
        self.beta_heatmap_data = self.qo_heatmap_data.copy()
        self.beta_heatmap = ValueHeatMap(
            data_shape,
            name="β(x)",
            cmap="YlGnBu",
            vmin=0.0,
            value_annon="Option",
            cbar_annon=True,
            reverse_y=True,
            **self.nrowcols,
        )
        if self._vis_value:
            self.value_heatmap = ValueHeatMap(
                data_shape,
                name="Qo",
                vmin=self.vmin,
                vmax=self.vmax,
                reverse_y=True,
                **self.nrowcols,
            )
        if self._vis_traj:
            self.trajectory = Trajectory(self.noptions)

        obs_space = env.observation_space
        obs_range = obs_space.high[:2] - obs_space.low[:2]
        obs_low = obs_space.low[:2]

        def point(state: np.ndarray) -> np.ndarray:
            xy = state[:2].copy()
            return (xy - obs_low) / obs_range

        self._point = point

        act_range = env.action_space.high - env.action_space.low
        act_low = env.action_space.low

        def scale_action(action: np.ndarray) -> np.ndarray:
            return act_range / (1.0 + np.exp(-action)) + act_low

        self._scale_action = scale_action
        self._quivers = [None] * self.noptions
        self._quiverkey = None

    def reset(
        self, agent: rainy.agents.Agent, env: rainy.envs.EnvExt, initial_state: State
    ) -> None:
        if self.counter == 0:
            self._initialize(env.unwrapped)

        if self._vis_traj:
            self.trajectory.reset(self._point(initial_state))

        with torch.no_grad():
            # This * is for making this compatible with rainy.agent.PPOCAgent
            pi, qo, beta, *_ = agent.net(self.states.view(-1, self.states.size(-1)))

        if self._vis_value:
            qo = qo.view(*self.states.shape[:2], self.noptions).mean(dim=1)
            qo = self.qo_heatmap_data.update(qo.cpu().numpy())
            for i in range(self.noptions):
                self.value_heatmap.update(qo[i, :], i)

        beta = beta.dist.probs.view(*self.states.shape[:2], self.noptions).mean(dim=1)
        beta = self.beta_heatmap_data.update(beta.cpu().numpy())
        for i in range(self.noptions):
            self.beta_heatmap.update(beta[i, :], i)

        if self._vis_policy:
            self.draw_policy(pi.dist.loc)

        # if self.counter == 0:
        #     self.beta_heatmap.fig.savefig("mjmaze-β-logp0.5to0.1-entm0.005.pdf")

        self.counter += 1

    def draw_policy(self, pimu: torch.Tensor) -> None:
        """ Visualize option-policies by two-dimentional arrows.
        """
        pimu = pimu.view(*self.states.shape[:2], self.noptions, -1)
        pimu = self._scale_action(pimu.cpu().numpy())
        ori = self.states[:, :, self._ori_index].cpu().unsqueeze(-1).numpy()
        ori = pimu[:, :, :, self._rot_index] + ori
        gain = pimu[:, :, :, self._gain_index]
        xvec = (np.cos(ori) * gain).mean(axis=1)
        yvec = (np.sin(ori) * gain).mean(axis=1)
        quiverkey_idx = self.noptions - 1 if self._is_flat else 0
        for opt in range(self.noptions):
            if self._quivers[opt] is not None:
                self._quivers[opt].remove()
            ax = self.beta_heatmap.axes[opt]
            self._quivers[opt] = ax.quiver(
                self._xy[:, 0],
                self._xy[:, 1],
                xvec[:, opt] * 2.0,
                yvec[:, opt] * 2.0,
                units="x",
                scale=0.5,
                width=0.06,
                headwidth=4,
                headlength=6,
            )
            if opt == quiverkey_idx:
                if self._quiverkey is not None:
                    self._quiverkey.remove()
                self._quiverkey = ax.quiverkey(
                    self._quivers[opt],
                    0.0,
                    1.05,
                    1,
                    "Expected value of Policy",
                    labelpos="E",
                )
        self.beta_heatmap.redraw()
        # self.beta_heatmap.axes[0].text(1.0, 17.8, "PPIMOC", fontsize=20)
        # self.beta_heatmap.fig.savefig(
        #     "ppimoc-pi-and-beta.pdf", bbox_inches="tight", pad_inches=0.0,
        # )

    def step(self, _env, _action, transition, net_outputs) -> None:
        if not self._vis_traj or "options" not in net_outputs:
            return

        if transition.terminal:
            self.trajectory.render()
        else:
            option = net_outputs["options"][0].item()
            self.trajectory.append(option, self._point(transition.state))
