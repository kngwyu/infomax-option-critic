""" Trajectory visualizer and heatmap.
"""

from typing import Dict, List, Tuple

import numpy as np
from rlpy.tools import plt

Point = np.ndarray

_N_ROW_COLS = {
    1: {"nrows": 1, "ncols": 1},
    2: {"nrows": 1, "ncols": 2},
    3: {"nrows": 2, "ncols": 2},
    4: {"nrows": 2, "ncols": 2},
    5: {"nrows": 2, "ncols": 3},
    6: {"nrows": 2, "ncols": 3},
    7: {"nrows": 2, "ncols": 4},
    8: {"nrows": 2, "ncols": 4},
}


def get_n_row_cols(noptions: int, is_flat_layout: bool = False) -> Dict[str, int]:
    if is_flat_layout:
        return {"nrows": 1, "ncols": noptions}
    else:
        return _N_ROW_COLS[noptions]


class ValueHeatMap:
    def __init__(
        self,
        data_shape: Tuple[int, int],
        nrows: int = 1,
        ncols: int = 1,
        name: str = "Value Function",
        cmap: str = "ValueFunction-New",
        vmin: float = -1.0,
        vmax: float = 1.0,
        value_annon: str = "Value",
        cbar_annon: bool = False,
        reverse_y: bool = False,
    ) -> None:
        scale_w = np.sqrt(ncols / nrows) * 1.1
        scale_h = np.sqrt(nrows / ncols) * 1.4
        w, h = plt.rcParams.get("figure.figsize")
        self.fig = plt.figure(name, figsize=(w * scale_w, h * scale_h))
        self.data_shape = data_shape
        cmap = plt.get_cmap(cmap)
        self.imgs = []
        self.axes = []

        def label_str(i: int) -> str:
            if nrows == 0 and ncols == 0:
                return ""
            else:
                return f"{value_annon} {i}"

        dummy = np.zeros(data_shape)
        for i in range(nrows * ncols):
            ax = self.fig.add_subplot(nrows, ncols, i + 1)
            img = ax.imshow(
                dummy, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(label_str(i))
            ax.set_aspect("equal")
            if reverse_y:
                ylim = ax.get_ylim()
                ax.set_ylim([ylim[1], ylim[0]])
            self.imgs.append(img)
            self.axes.append(ax)
            if i + 1 == ncols:
                from mpl_toolkits.axes_grid1 import make_axes_locatable

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="6%", pad=0.1)
                self.fig.colorbar(img, cax=cax, orientation="vertical")
                if cbar_annon:
                    cax.set_ylabel(name, rotation=0, position=(1.0, 0.55), fontsize=18)
                else:
                    cax.set_ylabel("")

        self.fig.tight_layout()
        self.fig.canvas.draw()

    def update(self, data, index: int = 0) -> None:
        self.imgs[index].set_data(data.reshape(self.data_shape))
        self.fig.canvas.draw()

    def redraw(self) -> None:
        self.fig.canvas.draw()


class Trajectory:
    COLORS: List[str] = [
        "xkcd:bright blue",
        "xkcd:neon green",
        "xkcd:bright purple",
        "xkcd:light orange",
        "xkcd:light red",
        "xkcd:cyan",
    ]

    def __init__(self, noptions: int, name: str = "Trajectory") -> None:
        self.noptions = noptions
        self.traj: List[Tuple[int, Point]] = []
        self.fig = plt.figure(name)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal")
        self.legend = None
        self.fig.tight_layout()
        self.lines = []

    def reset(self, initial: Point) -> None:
        self.traj = [(0, initial)]

    def append(self, option: int, point: Point) -> None:
        self.traj.append((option, point))

    def _clear_lines(self) -> None:
        for _ in range(len(self.lines)):
            self.lines.pop().remove()
        self.ax.clear()
        self.ax.set_autoscale_on(False)
        for i in range(self.noptions):
            self.ax.plot(
                [0.0], [0.0], color=self.COLORS[i], label=f"Option: {i}",
            )
        self.ax.legend(fontsize=16, loc="upper right", bbox_to_anchor=(1.2, 1.0))
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def render(self) -> None:
        self._clear_lines()
        self.traj.append((self.noptions, self.traj[-1][1]))
        current_opt = 0
        points = []
        for opt, point in self.traj:
            points.append(point)
            if current_opt == opt:
                continue
            points_arr = np.stack(points)
            color = self.COLORS[current_opt]
            points = [point]
            current_opt = opt
            if points_arr.ndim == 1:
                self.ax.plot(
                    [points_arr[0]], [points_arr[1]], color=color, marker="<",
                )
                continue
            self.ax.plot(points_arr[0, 0], points_arr[0, 1], color=color, marker="<")
            self.ax.plot(
                points_arr[:, 0], points_arr[:, 1], color=color, linewidth=3,
            )
            self.ax.plot(
                points_arr[-1, 0], points_arr[-1, 1], color=color,
            )
        self.fig.canvas.draw()
        # self.fig.savefig("ppimoc-traj.pdf")


if __name__ == "__main__":
    DATA_SHAPE = (20, 10)
    heatmap = ValueHeatMap(DATA_SHAPE, 2, 2)
    for i in range(4):
        heatmap.update(np.random.uniform(-1, 1, DATA_SHAPE), i)
    heatmap.draw()
    plt.show()
