from gridworld import Gridworld, RGBFullObsTransform
import tqdm
import torch
from collections import defaultdict

import torch.nn as nn
from torch.optim import Adam
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torch.nn.functional import log_softmax
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.envs import (
    StepCounter,
    RewardSum,
    TransformedEnv,
    FlattenObservation,
    CatTensors
)
import wandb
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import make_grid

wandb.init(project='grid_ppo_test')

env_batch_size = 64
frames_per_batch = env_batch_size * 2
total_frames = env_batch_size * 10

pbar = tqdm.tqdm(range(total_frames // env_batch_size))
logs = defaultdict(list)
device = 'cuda'
# For a complete training, bring the number of frames up to 1M
sub_batch_size = env_batch_size  # cardinality of the sub-samples gathered from the current data in the inner loop
ppo_steps_per_batch = 1  # optimisation steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 0.01
max_grad_norm = 1.0
hidden_dim = 128

env = Gridworld(batch_size=torch.Size([env_batch_size]), device=device)
check_env_specs(env)
env = TransformedEnv(
    env
)

env.append_transform(
    FlattenObservation(-2, -1,
                       in_keys=["player_tiles", "wall_tiles", "reward_tiles"],
                       out_keys=["flat_player_tiles", "flat_wall_tiles", "flat_reward_tiles"]
                       )
)

check_env_specs(env)

env.append_transform(
    CatTensors(
        in_keys=["flat_player_tiles", "flat_wall_tiles", "flat_reward_tiles"],
        out_key='flat_obs'
    )
)

check_env_specs(env)

env.append_transform(StepCounter())
env.append_transform(RewardSum(reset_keys=['_reset']))
env.append_transform(RGBFullObsTransform(5, in_keys=['wall_tiles'], out_keys=['image']))
check_env_specs(env)

in_features = env.observation_spec['flat_obs'].shape[-1]
actions_n = env.action_spec.n


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
    module=Value(in_features=in_features, hidden_dim=hidden_dim),
    in_keys=['flat_obs']
).to(device)


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


policy_net = Policy(in_features, hidden_dim, actions_n)

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
).to(device)

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    critic_coef=1.0,
    loss_critic_type="smooth_l1"
)

optim = Adam(loss_module.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

logs = defaultdict(list)
pbar = tqdm.tqdm(total=total_frames)
eval_str = ""


for i, tensordict_data in enumerate(collector):

    advantage_module(tensordict_data)
    loss_vals = loss_module(tensordict_data)
    loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
    )

    loss_value.backward()
    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
    optim.step()
    optim.zero_grad()


    def retrieve_episode_stats(tensordict_data, prefix=None):
        prefix = '' if prefix is None else f"{prefix}_"
        episode_reward = tensordict_data["next", "episode_reward"]
        step_count = tensordict_data["step_count"]
        state_value = tensordict_data['state_value']

        return {
            f"{prefix}episode_reward_mean": episode_reward.mean().item(),
            f"{prefix}episode_reward_max": episode_reward.max().item(),
            f"{prefix}step_count_max": step_count.max().item(),
            f"{prefix}state_value_max": state_value.max().item(),
            f"{prefix}state_value_mean": state_value.mean().item(),
            f"{prefix}state_value_min": state_value.min().item()
        }


    epi_stats = retrieve_episode_stats(tensordict_data, 'train')
    if i % 100:
        wandb.log(epi_stats, step=i)

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    logs["episode_reward"].append(epi_stats['train_episode_reward_mean'])
    pbar.update(tensordict_data.numel())

    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )

    logs["step_count"].append(epi_stats['train_step_count_max'])

    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"

    if i % 1024 == 0:
        with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
            eval_rollout = env.rollout(1000, policy_module)
            advantage_module(eval_rollout)
            epi_stats = retrieve_episode_stats(eval_rollout, prefix='eval')
            wandb.log(epi_stats, step=i)

            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    scheduler.step()

with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
    eval_rollout = env.rollout(1000, policy_module, break_when_any_done=False)
    eval_rollout = eval_rollout.cpu()
    observation = eval_rollout['image']
    walls = eval_rollout['wall_tiles']

    fig, ax = plt.subplots(1)
    img_plt = ax.imshow(make_grid(observation[:, 0]).permute(1, 2, 0))

    def animate(i):
        global text_plt
        x = make_grid(observation[:, i]).permute(1, 2, 0)
        img_plt.set_data(x)
        return

    myAnimation = animation.FuncAnimation(fig, animate, frames=90, interval=500, blit=False, repeat=False)

    # FFwriter = animation.FFMpegWriter(fps=1)
    # myAnimation.save('animation.mp4', writer=FFwriter)
    plt.show()