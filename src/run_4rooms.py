""" Run Gridworld experiments.
"""

import os
from typing import Optional

import click
from torch import optim

import a2imoc
import nets
import our_oc
import ppimoc
import rainy
import vis_4rooms

from rainy.envs import MultiProcEnv, RLPyGridWorld
from rainy.lib.explore import EpsGreedy
from rainy.net import termination_critic as tc

RLPyGridWorld.ALIASES.update(
    {
        "NoRooms": "RLPyFixedRewardGridWorld12x15-NoRooms",
        "Corridor": "RLPyFixedRewardGridWorld13x13-Corridor",
        "Passage": "RLPyFixedRewardGridWorld11x11-Passage",
        "NoRooms2": "RLPyFixedRewardGridWorld12x15-NoRooms2",
        "4RoomsNoRew": "RLPyFixedRewardGridWorld11x11-4Rooms-NoReward",
        "9RoomsEqual": "RLPyFixedRewardGridWorld17x17-9Rooms-Equal",
        "5Rooms1": "RLPyGridWorld13x13-5Rooms1",
        "5Rooms2": "RLPyGridWorld13x13-5Rooms2",
        "5Rooms3": "RLPyGridWorld13x13-5Rooms3",
    }
)


def select_agent(agent_name: str, **kwargs) -> rainy.agents.Agent:
    AGENTS = {
        "a2imoc": a2imoc.A2ImocAgent,
        "our-aoc": our_oc.OurAOCAgent,
        "ppimoc": ppimoc.PPImocAgent,
        "our-ppoc": our_oc.OurPPOCAgent,
        "a2c": rainy.agents.A2CAgent,
        "actc": rainy.agents.ACTCAgent,
        "aoc": rainy.agents.AOCAgent,
        "ppo": rainy.agents.PPOAgent,
    }
    return AGENTS[agent_name]


@rainy.subcommand()
@click.argument("new-envname", type=str)
@click.option("--comment", type=str, default=None)
@click.option("--logdir", type=str, default=None)
@click.option("--eval-render", is_flag=True)
def train_and_adapt(
    ctx: click.Context,
    new_envname: str,
    comment: Optional[str],
    logdir: Optional[str],
    eval_render: bool,
) -> None:
    new_envs = new_envname.split("/")
    experiment = ctx.obj.experiment
    script_path = ctx.obj.script_path
    if script_path is not None:
        fingerprint = dict(
            comment="" if comment is None else comment, kwargs=ctx.obj.kwargs,
        )
        experiment.logger.setup_from_script_path(
            script_path, dirname=logdir, fingerprint=fingerprint,
        )
    cfg = experiment.config
    cfg.keep_logger = True
    experiment.train(eval_render=eval_render)
    for new_env in new_envs:
        cfg.eval_env.close()
        cfg.keep_logger = False
        # Set new environments
        obs_type = cfg.eval_env.obs_type
        cfg.set_env(lambda: RLPyGridWorld(new_env, obs_type=obs_type))
        cfg.eval_env = cfg.env()
        experiment.ag.penv = cfg.parallel_env()
        experiment.ag.eval_penv = None
        experiment._retrain_impl(cfg.max_steps, eval_render=eval_render)


@rainy.main(script_path=os.path.realpath(__file__), agent_selector=select_agent)
@rainy.option("--visualize-beta", "-VB", is_flag=True)
@rainy.option("--visopt-beta-pi", is_flag=True)
@rainy.option("--visopt-flat", is_flag=True)
@rainy.option("--not-upgoing", "-NU", is_flag=True)
def main(
    envname: str = "4RoomsExp",
    max_steps: int = 600000,
    obs_type: str = "image",
    num_options: int = 4,
    visualize_beta: bool = False,
    visopt_beta_pi: bool = False,
    visopt_flat: bool = False,
    logmu_weight: float = 0.5,
    logmu_weight_min: Optional[float] = None,
    beta_loss_weight: float = 1.0,
    entropy_weight: float = 0.01,
    pimu_entropy_weight: float = 0.04,
    agent_name: str = "a2imoc",
    option_selector: str = "logp",
    adv_type: str = "upgoing",
    opt_delib_cost: float = 0.0,
    ext_adv_weight: float = 1.0,
    int_adv_weight: float = 1.0,
    eval_times: int = 1,
    eval_freq: int = 10000,
    **kwargs,
) -> rainy.Config:
    assert obs_type in ["image", "binary-image"], f"Invalid obs: {obs_type}"
    c = rainy.Config()
    if visualize_beta:
        if agent_name in ["a2imoc", "our-aoc", "ppimoc", "our-ppoc"]:
            c.eval_hooks.append(
                vis_4rooms.ImocVis(
                    num_options, is_flat_layout=visopt_flat, beta_pi=visopt_beta_pi,
                )
            )
        elif agent_name in ["actc"]:
            c.eval_hooks.append(vis_4rooms.ActcVis(num_options))
        elif agent_name == "aoc":
            c.eval_hooks.append(
                vis_4rooms.OCVis(
                    num_options, is_flat_layout=visopt_flat, beta_pi=visopt_beta_pi,
                )
            )
        else:
            vis_cls = vis_4rooms.A2CVis
            c.eval_hooks.append(vis_cls())
    c.set_env(lambda: RLPyGridWorld(envname, obs_type))
    c.set_parallel_env(MultiProcEnv)
    c.max_steps = max_steps
    c.adv_type = adv_type

    # Algorithm specific configurations
    if agent_name in ["a2imoc", "a2c", "our-aoc", "actc", "aoc"]:
        c.nworkers = 12
        c.nsteps = 20
        c.set_optimizer(lambda params: optim.RMSprop(params, lr=2e-3))
    elif agent_name in [
        "ppimoc",
        "ppo",
        "our-ppoc",
    ]:
        c.nsteps = 32
        c.ppo_minibatch_size = 32
        c.nworkers = 8
        c.ppo_epochs = 4
        c.ppo_clip = 0.1
        c.beta_logit_clip = 0.1
        c.set_optimizer(lambda params: optim.Adam(params, lr=4e-4))
        c.adv_normalize_eps = None
        c.use_gae = True
    else:
        raise NotImplementedError(f"NotImplemented agent: {agent_name}")

    # Option parameters
    c.option_selector = option_selector
    c.logmu_weight = logmu_weight
    c.logmu_weight_min = logmu_weight_min
    c.set_explorer(lambda: EpsGreedy(0.1))
    c.set_explorer(lambda: EpsGreedy(0.01), key="eval")
    # For A2C-like methods, 480 and 240 are used.
    c.opt_model_capacity = c.nworkers * c.nsteps * 2
    c.opt_model_batch_size = c.opt_model_capacity // 2
    # For AOC
    c.opt_delib_cost = opt_delib_cost

    # loss weights
    c.beta_loss_weight = beta_loss_weight
    c.value_loss_weight = 1.0
    c.grad_clip = 1.0
    c.eval_freq = eval_freq
    c.network_log_freq = (c.max_steps // c.batch_size) // 10
    c.entropy_weight = entropy_weight
    c.pimu_entropy_weight = pimu_entropy_weight
    c.eval_times = eval_times

    CONV_ARGS = dict(
        hidden_channels=(16, 16), feature_dim=128, cnn_params=[(4, 1), (2, 1)],
    )
    if agent_name == "actc":
        c.set_net_fn(
            "actor-critic", tc.oac_conv_shared(num_options=num_options, **CONV_ARGS),
        )
    else:
        c.set_net_fn("actor-critic", rainy.net.actor_critic.conv_shared(**CONV_ARGS))
    c.set_net_fn(
        "option-critic", nets.conv_shared(num_options=num_options, **CONV_ARGS),
    )
    c.set_optimizer(lambda params: optim.Adam(params, lr=1e-4), key="termination")
    c.set_net_fn(
        "termination-critic", tc.tc_conv_shared(num_options=num_options, **CONV_ARGS),
    )
    return c


if __name__ == "__main__":
    main()
