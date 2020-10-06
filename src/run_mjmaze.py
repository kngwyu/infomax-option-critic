""" Run MuJoCo Maze experiments.
"""

import os

from typing import Optional

import click
import numpy as np

from torch import optim

import nets
import ppimoc
import our_oc
import rainy
import vis_mjmaze

from rainy.envs import EnvExt, pybullet_parallel
from rainy.net import option_critic as oc
from rainy.net.policy import PerOptionStdGaussianDist, SeparateStdGaussianDist


BILLIARD_GOALS = [(2.0, -3.0), (-2.0, -3.0), (-2.0, 1.0), (2.0, 1.0)]


def _change_billiard_goal(primary: int) -> dict:
    return {
        "primary_goal": BILLIARD_GOALS[primary],
        "subgoals": BILLIARD_GOALS[:primary] + BILLIARD_GOALS[primary + 1 :],
    }


class MazeEnvExt(EnvExt):
    CUSTOM_ENVS = {
        "PointTRoom2": ("PointTRoom-v1", {"goal": (-2.0, -3.0)}),
        "PointTRoomSub2": (
            "PointTRoom-v2",
            {"primary_goal": (-2.0, -3.0), "subgoal": (2.0, -3.0)},
        ),
        "PointBilliard2": ("PointBilliard-v2", _change_billiard_goal(1)),
        "PointBilliard3": ("PointBilliard-v2", _change_billiard_goal(2)),
        "PointBilliard4": ("PointBilliard-v2", _change_billiard_goal(3)),
    }

    def __init__(self, name: str) -> None:
        import gym
        import mujoco_maze  # noqa

        if name in self.CUSTOM_ENVS:
            name, task_kwargs = self.CUSTOM_ENVS[name]
            super().__init__(gym.make(name, task_kwargs=task_kwargs))
        else:
            if "Swimmer" in name or "Reacher" in name:
                super().__init__(gym.make(name, forward_reward_weight=0.0))
            else:
                super().__init__(gym.make(name))
        self.action_shift = self.action_space.low
        self.action_scale = self.action_space.high - self.action_space.low

    def step(self, action):
        action = self.action_scale / (1.0 + np.exp(-action)) + self.action_shift
        return super().step(action)


def select_agent(agent_name: str, **kwargs) -> rainy.agents.Agent:
    AGENTS = {
        "ppimoc": ppimoc.PPImocAgent,
        "ppo": rainy.agents.PPOAgent,
        "ppoc": rainy.agents.PPOCAgent,
        "our-ppoc": our_oc.OurPPOCAgent,
    }
    return AGENTS[agent_name]


@rainy.subcommand()
@click.argument("new-envname", type=str)
@click.option("--comment", type=str, default=None)
@click.option("--logdir", type=str, default=None)
@click.option("--additional-steps", type=int, default=None)
@click.option("--eval-render", is_flag=True)
def train_and_adapt(
    ctx: click.Context,
    new_envname: str,
    comment: Optional[str],
    logdir: Optional[str],
    additional_steps: Optional[int],
    eval_render: bool,
) -> None:
    new_envs = new_envname.split("/")
    click.secho(f"adapted envs: {new_envs}", fg="red")
    experiment = ctx.obj.experiment
    script_path = ctx.obj.script_path
    if script_path is not None:
        fingerprint = dict(
            comment="" if comment is None else comment, kwargs=ctx.obj.kwargs,
        )
        experiment.logger.setup_from_script_path(
            script_path, dirname=logdir, fingerprint=fingerprint
        )
    cfg = experiment.config
    cfg.keep_logger = True
    experiment.train(eval_render=eval_render)
    for i, new_env in enumerate(new_envs):
        cfg.eval_env.close()
        if i + 1 == len(new_envs):
            cfg.keep_logger = False
        # Set new environments
        cfg.set_env(lambda: MazeEnvExt(new_env))
        cfg.eval_env = MazeEnvExt(new_env)
        experiment.ag.penv = cfg.parallel_env()
        experiment.ag.eval_penv = None
        additional_steps = additional_steps or cfg.max_steps
        if cfg.logmu_weight_min is not None:
            experiment.ag._opt_logp_cooler = rainy.lib.explore.LinearCooler(
                cfg.logmu_weight,
                cfg.logmu_weight_min,
                additional_steps // cfg.nworkers,
            )
        experiment._retrain_impl(additional_steps, eval_render=eval_render)


@rainy.main(script_path=os.path.realpath(__file__), agent_selector=select_agent)
@rainy.option("--visualize-beta", "-VB", is_flag=True)
@rainy.option("--visualize-policy", "-VP", is_flag=True)
@rainy.option("--use-separated-network", "-SN", is_flag=True)
@rainy.option("--not-upgoing", "-NU", is_flag=True)
def main(
    envname: str = "PointUMaze-v1",
    max_steps: int = 4,
    num_options: int = 4,
    visualize_beta: bool = False,
    visualize_policy: bool = False,
    normalize_reward: bool = False,
    entropy_weight: float = 0.001,
    pimu_entropy_weight: float = 0.004,
    logmu_weight: float = 0.4,
    logmu_weight_min: Optional[float] = None,
    beta_logit_clip: float = 0.1,
    beta_loss_weight: float = 1.0,
    pimu_mc_rollout: int = 20,
    adv_type: str = "upgoing",
    option_selector: str = "logp",
    agent_name: str = "ppimoc",
    use_separated_network: bool = False,
    not_upgoing: bool = False,
    eval_times: int = 4,
) -> rainy.Config:
    c = rainy.Config()
    if visualize_beta or visualize_policy:
        if agent_name in ["ppimoc", "our-ppoc", "ppoc"]:
            c.eval_hooks.append(
                vis_mjmaze.ImocVis(num_options, vis_policy=visualize_policy)
            )
        elif agent_name == "ppo":
            c.eval_hooks.append(vis_mjmaze.A2CVis())
        else:
            raise NotImplementedError("Visualizer for PPOC is not yet implemented")
    if max_steps < 20:
        max_steps *= int(1e6)
    c.max_steps = max_steps
    # Environment settings
    c.set_env(lambda: MazeEnvExt(envname))
    c.eval_env = MazeEnvExt(envname)
    c.discount_factor = 0.99
    c.adv_type = adv_type
    c.set_parallel_env(
        pybullet_parallel(normalize_obs=False, normalize_reward=normalize_reward)
    )

    # Algorithm specific configurations
    if "pp" in agent_name:
        c.nworkers = 16
        c.nsteps = 256
        c.ppo_minibatch_size = (c.nworkers * c.nsteps) // 4
        c.ppo_epochs = 10
        c.set_optimizer(lambda params: optim.Adam(params, lr=3e-4, eps=1e-4))
        c.adv_normalize_eps = None
        c.ppo_clip = 0.2
        c.use_gae = True
    else:
        raise NotImplementedError(f"NotImplemented agent: {agent_name}")

    # Option parameters
    c.option_selector = option_selector
    c.logmu_weight = logmu_weight
    c.logmu_weight_min = logmu_weight_min
    c.opt_model_capacity = c.nworkers * c.nsteps
    c.opt_model_batch_size = c.opt_model_capacity // 2
    c.set_explorer(lambda: rainy.lib.explore.EpsGreedy(0.1))
    c.set_explorer(lambda: rainy.lib.explore.EpsGreedy(0.1), key="eval")
    c.grad_clip = 0.5
    c.pimu_mc_rollout = pimu_mc_rollout
    if not_upgoing:
        c.upgoing_adv = False
    # optimization parameters
    c.entropy_weight = entropy_weight
    c.pimu_entropy_weight = pimu_entropy_weight
    c.value_loss_weight = 1.0
    c.beta_loss_weight = beta_loss_weight
    c.beta_logit_clip = beta_logit_clip

    c.set_net_fn(
        "actor-critic",
        rainy.net.actor_critic.fc_shared(policy=SeparateStdGaussianDist),
    )
    if agent_name == "ppoc":
        c.set_net_fn(
            "option-critic",
            oc.fc_shared(
                num_options=num_options, policy=PerOptionStdGaussianDist, has_mu=True,
            ),
        )
    else:
        if use_separated_network:
            c.set_net_fn(
                "option-critic",
                nets.fc_separated(
                    num_options=num_options, policy=PerOptionStdGaussianDist,
                ),
            )
        else:
            c.set_net_fn(
                "option-critic",
                nets.fc_shared(
                    num_options=num_options, policy=PerOptionStdGaussianDist,
                ),
            )
    c.episode_log_freq = 100
    c.network_log_freq = 10
    c.eval_times = eval_times
    c.eval_freq = c.max_steps // 50
    return c


if __name__ == "__main__":
    main()
