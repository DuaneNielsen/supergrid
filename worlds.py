import torch
from torch import tensor
from tensordict import TensorDict
from gridworld import Gridworld, pos_to_grid, RGBFullObsTransform
from torchrl.envs import (
    EnvBase,
    Resize,
    ToTensorImage,
    PermuteTransform,
    StepCounter,
    RewardSum,
    TransformedEnv,
    FlattenObservation,
    CatTensors
)


def gen_maze(batch_size=None):
    """

    To change the layout of the gridworld, change these parameters

    walls: 1 indicates the position of a wall.  The boundary grid cells must have a wall.
    rewards: The amount of reward for entering the tile.
        Rewards are only received the first time the agent enters the tile
    terminal_states: Indicated by 1, when this tile is entered, the terminated flag is set true

    :param batch_size: the number of environments to run simultaneously
    :return: a batch_size tensordict, with the following entries

       "player_pos": N, 2 tensor indices that correspond to the players location
       "player_tiles": N, 5, 5 tensor, with a single tile set to 1 that indicates player position
       "wall_tiles": N, 5, 5 tensor, 1 indicates wall
       "reward_tiles": N, 5, 5 tensor, rewards remaining in environment
       "terminal_tiles": N, 5, 5 tensor, episode will terminate when tile with value True is entered

    """
    walls = tensor([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ], dtype=torch.float32)

    rewards = tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, -1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32)

    terminal_states = tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.bool)

    H, W = walls.shape
    player_pos = tensor([2, 2], dtype=torch.int64)
    player_tiles = pos_to_grid(player_pos, H, W, dtype=walls.dtype)

    state = {
        "player_pos": player_pos,
        "player_tiles": player_tiles,
        "wall_tiles": walls,
        "reward_tiles": rewards,
        "terminal_tiles": terminal_states
    }

    td = TensorDict(state, batch_size=[])

    if batch_size:
        td = td.expand(batch_size).contiguous()
    return td


class MazeWorld(Gridworld):
    gen_params = staticmethod(gen_maze)

if __name__ == '__main__':

    """
    Optimize the agent using Proximal Policy Optimization (Actor - Critic)
    the Generalized Advantage Estimation module is used to compute Advantage
    """

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--device', default='cpu', help="cuda or cpu")
    parser.add_argument('--env_batch_size', type=int, default=16, help="number of environments")
    parser.add_argument('--steps_per_batch', type=int, default=32, help="number of steps to take in env per batch")
    parser.add_argument('--train_steps', type=int, default=1000, help="number of PPO updates to run")
    parser.add_argument('--clip_epsilon', type=float, default=0.1, help="PPO clipping parameter")
    parser.add_argument('--gamma', type=float, default=0.95, help="GAE gamma parameter")
    parser.add_argument('--lmbda', type=float, default=0.9, help="GAE lambda parameter")
    parser.add_argument('--entropy_eps', type=float, default=0.001, help="policy entropy bonus weight")
    parser.add_argument('--max_grad_norm', type=float, default=2.0, help="gradient clipping")
    parser.add_argument('--hidden_dim', type=int, default=64, help="hidden dim size of MLP")
    parser.add_argument('--lr', type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument('--eval_freq', type=int, default=256, help="run eval after this many training steps")
    parser.add_argument('--demo', action='store_true', help="command switch to visualize after training completes")
    parser.add_argument('--wandb', action='store_true', help='command switch to enable wandb logging')
    args = parser.parse_args()

    from collections import defaultdict
    from statistics import mean
    import tqdm
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from tensordict.nn import TensorDictModule
    from torch.distributions import Categorical
    from torch.nn.functional import log_softmax
    from torchrl.modules import ProbabilisticActor, ValueOperator
    from torchrl.collectors import SyncDataCollector
    from torchrl.objectives import ClipPPOLoss
    from torchrl.objectives.value import GAE
    from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
    from torchrl.record.loggers import CSVLogger
    from torchrl.record import VideoRecorder
    from hrid import HRID

    from matplotlib import pyplot as plt
    import matplotlib
    import matplotlib.animation as animation

    matplotlib.use('QtAgg')
    from torchvision.utils import make_grid

    exp_name = HRID().generate()
    logger = CSVLogger(exp_name, 'csv', video_format='mp4', video_fps=3)

    frames_per_batch = args.env_batch_size * args.steps_per_batch
    total_frames = args.env_batch_size * args.steps_per_batch * args.train_steps

    env = MazeWorld(batch_size=torch.Size([args.env_batch_size]), device=args.device)
    env = TransformedEnv(
        env
    )
    env.append_transform(
        FlattenObservation(-2, -1,
                           in_keys=["player_tiles", "wall_tiles", "reward_tiles"],
                           out_keys=["flat_player_tiles", "flat_wall_tiles", "flat_reward_tiles"]
                           )
    )
    env.append_transform(
        CatTensors(
            in_keys=["flat_player_tiles", "flat_wall_tiles", "flat_reward_tiles"],
            out_key='flat_obs'
        )
    )
    env.append_transform(StepCounter())
    env.append_transform(RewardSum(reset_keys=['_reset']))
    env.append_transform(RGBFullObsTransform())
    check_env_specs(env)

    in_features = env.observation_spec['flat_obs'].shape[-1]
    actions_n = env.action_spec.n

    # value function to compute advantage
    class Value(nn.Module):
        def __init__(self, in_features, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=1, bias=False)
            )

        def forward(self, obs):
            values = self.net(obs)
            return values

    value_module = ValueOperator(
        module=Value(in_features=in_features, hidden_dim=args.hidden_dim),
        in_keys=['flat_obs']
    ).to(args.device)

    # policy network
    class Policy(nn.Module):
        def __init__(self, in_features, hidden_dim, actions_n):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=hidden_dim),
                nn.ReLU(),
                nn.Linear(in_features=hidden_dim, out_features=actions_n, bias=False)
            )

        def forward(self, obs):
            return log_softmax(self.net(obs), dim=-1)

    policy_net = Policy(in_features, args.hidden_dim, actions_n)

    policy_module = TensorDictModule(
        policy_net,
        in_keys=["flat_obs"],
        out_keys=["logits"],
    )

    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=['logits'],
        out_keys=['action'],
        distribution_class=Categorical,
        return_log_prob=True
    ).to(args.device)

    # no need to reuse data for PPO as it's an online algo
    # so we will go with datacollector only and collect fresh batches each time

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=args.device,
    )

    advantage_module = GAE(
        gamma=args.gamma, lmbda=args.lmbda, value_network=value_module, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=args.clip_epsilon,
        entropy_bonus=bool(args.entropy_eps),
        entropy_coef=args.entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1"
    )

    optim = Adam(loss_module.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, 0.0
    )

    def log_episode_stats(tensordict_data, key, prefix, i):
        terminal_values = tensordict_data['next', key][tensordict_data['next', 'done']].flatten().tolist()
        value_mean, value_max, value_n = None, None, None
        if len(terminal_values) > 0:
            value_mean, value_max, value_n = mean(terminal_values), max(terminal_values), len(terminal_values)
            logger.log_scalar(f"{prefix}_{key}_mean", value_mean, i)
            logger.log_scalar(f"{prefix}_{key}_max", value_max, i)
            logger.log_scalar(f"{prefix}_{key}_n", value_max, i)
        return value_mean, value_max, value_n

    # training loop starts here

    logs = defaultdict(list)
    pbar = tqdm.tqdm(total=total_frames)
    eval_str = ""
    train_reward_mean, train_reward_max, train_reward_n, = 0., 0., 0
    eval_reward_mean, eval_reward_max, eval_reward_n = 0., 0., 0

    for i, tensordict_data in enumerate(collector):

        train_reward_mean, train_reward_max, train_reward_n = (
            log_episode_stats(tensordict_data, "episode_reward", "train", i))
        log_episode_stats(tensordict_data, "step_count", "train", i)
        pbar.update(tensordict_data.numel())

        advantage_module(tensordict_data)
        loss_vals = loss_module(tensordict_data)
        loss_value = (loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"])
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad()

        logger.log_scalar('lr', scheduler.get_last_lr()[0])

        if train_reward_mean is not None and eval_reward_mean is not None:
            pbar.set_description(
                f'train reward mean/max (n) {train_reward_mean:.2f}/{train_reward_max:.2f} ({train_reward_n}) '
                f'eval reward mean/max (n): {eval_reward_mean:.2f}/{eval_reward_max:.2f} ({eval_reward_n})')

        if i % args.eval_freq == 0:
            with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
                eval_rollout = env.rollout(1000, policy_module, break_when_any_done=False)
                advantage_module(eval_rollout)
                eval_reward_mean, eval_reward_max, eval_reward_n = \
                    log_episode_stats(tensordict_data, "episode_reward", "eval", i)
                log_episode_stats(tensordict_data, "step_count", "eval", i)

        scheduler.step()

    if args.demo:
        with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
            pbar.set_description('rolling out video')
            N, H, W = env.observation_spec['wall_tiles'].shape
            env.append_transform(
                Resize(H * 8, W * 8, in_keys=['pixels'], out_keys=['pixels'], interpolation='nearest'))
            env.append_transform(PermuteTransform([-2, -1, -3], in_keys=['pixels'], out_keys=['pixels']))
            recorder = VideoRecorder(logger=logger, tag='gridworld', fps=3, skip=1)
            env.append_transform(recorder)
            eval_rollout = env.rollout(1000, policy_module, break_when_any_done=False)
            pbar.set_description(f'writing video to {exp_name}')
            recorder.dump()
            pbar.close()

            """
            The data dict layout is transitions

            {(S, A), next: {R_next, S_next, A_next}}

            [
              { state_t0, reward_t0, terminal_t0, action_t0 next: { state_t2, reward_t2:1.0, terminal_t2:False } },
              { state_t1, reward_t1, terminal_t1, action_t1 next: { state_t3, reward_t2:-1.0, terminal_t3:True } }  
              { state_t0, reward_t0, terminal_t0, action_t0 next: { state_t3, reward_t2:1.0, terminal_t3:False } }
            ]

            But which R to use, R or next: R?

            recall: Q(S, A) = R + Q(S_next, A_next)

            Observe that reward_t0 is always zero, reward is a consequence for taking an action in a state, therefore...

            reward = data['next']['reward'][timestep]

            Which terminal to use?

            Recall that the value of a state is the expectation of future reward.
            Thus the terminal state has no value, therefore...

            Q(S, A) = R_next + Q(S_next, A_next) * terminal_next

            terminal =  data['next']['terminal'][timestep]
            """

            eval_rollout = eval_rollout.cpu()
            observation = eval_rollout['pixels']
            walls = eval_rollout['wall_tiles']

            fig, ax = plt.subplots(1)
            img_plt = ax.imshow(make_grid(observation[:, 0].permute(0, 3, 1, 2)).permute(1, 2, 0))
            plt.title(f"supergrid {exp_name}")

            def animate(i):
                x = make_grid(observation[:, i].permute(0, 3, 1, 2)).permute(1, 2, 0)
                img_plt.set_data(x)
                return

            myAnimation = animation.FuncAnimation(fig, animate, frames=90, interval=500, blit=False, repeat=False)
            plt.show()
