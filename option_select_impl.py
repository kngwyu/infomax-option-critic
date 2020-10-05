""" Implemenation of uncertainty-aware option selection
"""


from abc import ABC, abstractmethod
from typing import Tuple

import torch

from torch import BoolTensor, LongTensor, Tensor
from torch.distributions import Categorical

from rainy.net.policy import BernoulliPolicy


def _debug_minmax(name: str, t: Tensor) -> None:
    print(f"{name}: {t.max().item()}, {t.min().item()}")


class OptionSelectImpl(ABC):
    worker_indices: Tensor
    EPS = 0.001
    INF = 1e9

    @abstractmethod
    def logmu_weight(self) -> float:
        pass

    def _logmu(self, qo: Tensor, logmu_o_xs: Tensor) -> Tensor:
        return qo - self.logmu_weight() * logmu_o_xs

    def _eval_sample_options(
        self, qo: Tensor, beta: BernoulliPolicy,
    ) -> Tuple[LongTensor, BoolTensor]:
        """Sample options by ε-Greedy
        """
        batch_size = qo.size(0)
        prev_options = self.eval_prev_options[:batch_size]
        current_beta = beta[self.worker_indices[:batch_size], prev_options]
        opt_terminals = current_beta.action().bool()
        use_new_options = self.eval_is_initial_states[:batch_size] | opt_terminals
        new_options = self.eval_opt_explorer.select_from_value(qo, same_device=True)
        options = torch.where(use_new_options, new_options, prev_options)
        return options, use_new_options

    def _sample_options(
        self, qo: Tensor, beta: BernoulliPolicy, mu_o_xs: Categorical,
    ) -> Tuple[LongTensor, BoolTensor]:
        """
        Select new options.
        Returns options and booltensor that indicates which options ended.
        """

        masks = self.storage.masks[-1]
        prev_options = self.prev_options
        current_beta = beta[self.worker_indices[: qo.size(0)], prev_options]
        opt_terminals = current_beta.action().bool()
        use_new_options = (1.0 - masks).bool() | opt_terminals
        # mask out current options
        opt_mask = torch.zeros_like(qo)
        opt_mask[self.worker_indices, prev_options] += opt_terminals * -self.INF
        if self.config.option_selector == "epsg":
            new_options = self.opt_explorer.select_from_value(
                qo + opt_mask, same_device=True
            )
        elif self.config.option_selector == "logp":
            new_options = self._logmu(qo + opt_mask, mu_o_xs.logits).argmax(-1)
        elif self.config.option_selector == "epsg-logp":
            value = self._logmu(qo + opt_mask, mu_o_xs.logits)
            new_options = self.opt_explorer.select_from_value(value, same_device=True)
        else:
            raise NotImplementedError(
                f"Invalid option selector {self.config.opt_selector}"
            )
        self.option_counter[new_options[use_new_options].cpu().numpy()] += 1
        options = torch.where(use_new_options, new_options, prev_options)
        return options, opt_terminals
