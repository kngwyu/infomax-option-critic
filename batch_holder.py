""" Replay Buffer for storing option transitions.
"""


from typing import Callable, Tuple

import numpy as np

from rainy.prelude import Array


class XsHolder:
    def __init__(
        self,
        xs: Array,
        options: Array,
        xs_xf_callback: Callable[[Array[float], Array[int], Array[bool]], None],
    ) -> None:
        self.xs = xs.copy()
        self.options = options.copy()
        self._xs_xf_callback = xs_xf_callback

    def update(
        self,
        states: Array[float],
        prev_terminals: Array[bool],
        options: Array[int],
        opt_terminals: Array[bool],
    ) -> None:
        """
        states: Si
        prev_terminals: Si == S0?
        options, opt_terminals: Sampled from P(O|Si)
        """
        # If the option or the episode end, let's update Xs to the new one.
        update = np.logical_or(opt_terminals, prev_terminals)
        # If the option ends and state is not S0, let's push it to the replay bufffer
        push = np.logical_and(opt_terminals, np.logical_not(prev_terminals))
        if push.any():
            self._xs_xf_callback(self.xs[push], states[push], self.options[push])
        self.xs[update] = states[update]
        self.options[update] = options[update]


class XsXfHolder:
    def __init__(
        self, num_options: int, state_shape: tuple, batch_size: int, capacity: int,
    ) -> None:
        self.xs_buf = np.zeros((num_options, capacity, *state_shape))
        self.xf_buf = np.zeros((num_options, capacity, *state_shape))
        self.next_idx = [0] * num_options
        self.capacity = capacity
        self.length = [0] * num_options
        self.batch_size = batch_size
        self.noptions = num_options

    def clear(self) -> None:
        for i in range(self.noptions):
            self.length[i] = 0
            self.next_idx[i] = 0
        self.xs_buf.fill(0.0)
        self.xf_buf.fill(0.0)

    def append(self, xs_: Array[float], xf_: Array[float], opt_: Array[int]) -> None:
        batch_size = xs_.shape[0]
        if batch_size == 0:
            return

        for xs, xf, opt, in zip(xs_, xf_, opt_):
            idx = self.next_idx[opt]
            self.xs_buf[opt, idx] = xs
            self.xf_buf[opt, idx] = xf
            self.next_idx[opt] = (idx + 1) % self.capacity
            self.length[opt] = min(self.capacity, self.length[opt] + 1)

    def ready(self) -> bool:
        each_opt_size = self.batch_size // self.noptions
        return all(map(lambda len_: len_ >= each_opt_size, self.length))

    def get_batch(self) -> Tuple[Array[float], Array[float]]:
        each_opt_size = self.batch_size // self.noptions
        xs, xf, = [], []
        for i in range(self.noptions):
            indices = np.random.randint(self.length[i], size=(each_opt_size,))
            for idx in indices:
                xs.append(self.xs_buf[i, idx])
                xf.append(self.xf_buf[i, idx])
        return np.stack(xs), np.stack(xf)

    def opt_target(self) -> Array[int]:
        opt = []
        each_opt_size = self.batch_size // self.noptions
        for i in range(self.noptions):
            opt += [i] * each_opt_size
        return np.array(opt)
