from typing import Optional

import torch
from torch import tensor
from tensordict import TensorDict
from torchrl.data import CompositeSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec, UnboundedDiscreteTensorSpec, DiscreteTensorSpec, \
    UnboundedContinuousTensorSpec
from torchrl.envs import (
    EnvBase,
    Transform,
    TransformedEnv,
    RewardSum,
)
from torchrl.envs.utils import check_env_specs, step_mdp
from torchrl.envs.transforms.transforms import _apply_to_composite, ObservationTransform, Resize
from torch.nn.functional import interpolate
from torchvision.utils import make_grid
from math import prod
from tensordict.utils import expand_as_right, expand_right, NestedKey
from typing import Any, Dict, List, Optional, OrderedDict, Sequence, Tuple, Union
from enum import IntEnum

"""
A minimal stateless vectorized gridworld in pytorch rl
Action space: (0, 1, 2, 3) -> N, E, S, W

Features

walls
1 time pickup rewards or penalties
terminated tiles
outputs a fully observable RGB image 

look at the gen_params function to setup the world

example of configuring and performing a rollout at bottom

Maybe I will write an even simpler stateless non-vectorized version of this gridworld
"""


class Actions(IntEnum):
    N = 0,
    E = 1,
    S = 2,
    W = 3


# N/S is reversed as y-axis in images is reversed
action_vec = [
    tensor([-1, 0]),  # N
    tensor([0, 1]),  # E
    tensor([1, 0]),  # S
    tensor([0, -1])  # W
]
action_vec = torch.stack(action_vec)

yellow = tensor([255, 255, 0], dtype=torch.uint8)
red = tensor([255, 0, 0], dtype=torch.uint8)
green = tensor([0, 255, 0], dtype=torch.uint8)
pink = tensor([255, 0, 255], dtype=torch.uint8)
violet = tensor([226, 43, 138], dtype=torch.uint8)
white = tensor([255, 255, 255], dtype=torch.uint8)
gray = tensor([128, 128, 128], dtype=torch.uint8)
light_gray = tensor([211, 211, 211], dtype=torch.uint8)
blue = tensor([0, 0, 255], dtype=torch.uint8)


def pos_to_grid(pos, H, W, device='cpu', dtype=torch.float32):
    """
    Converts positions to grid where 1 indicates a position
    :param pos: N, 2 tensor of grid positions (x = H, y = W) or 2 tensor
    :param H: height
    :param W: width
    :param: device: device
    :param: dtype: type of tensor
    :return: N, H, W tensor or single H, W tensor
    """

    if len(pos.shape) == 2:
        N = pos.size(0)
        batch_range = torch.arange(N, device=device)
        grid = torch.zeros((N, H, W), dtype=dtype, device=device)
        grid[batch_range, pos[:, 0], pos[:, 1]] = 1.
    else:
        grid = torch.zeros((H, W), dtype=dtype, device=device)
        grid[pos[0], pos[1]] = 1.
    return grid


def _step(state):

    device = state.device
    N, H, W = state['wall_tiles'].shape
    batch_range = torch.arange(N, device=device)
    dtype = state['wall_tiles'].dtype
    action = state['action'].squeeze(-1)

    # move player position checking for collisions
    direction = action_vec.to(device)
    next_player_pos = state['player_pos'] + direction[action]
    next_player_grid = pos_to_grid(next_player_pos, H, W, device=device, dtype=torch.bool)
    collide_wall = torch.logical_and(next_player_grid, state['wall_tiles'] == 1).any(-1).any(-1)
    player_pos = torch.where(collide_wall[..., None], state['player_pos'], next_player_pos)
    player_tiles = pos_to_grid(player_pos, H, W, device=device, dtype=dtype)

    # pickup any rewards
    reward = state['reward_tiles'][batch_range, player_pos[:, 0], player_pos[:, 1]]
    state['reward_tiles'][batch_range, player_pos[:, 0], player_pos[:, 1]] = 0.

    # set terminated flag if hit terminal tile
    terminated = state['terminal_tiles'][batch_range, player_pos[:, 0], player_pos[:, 1]]

    out = {
        'player_pos': player_pos,
        'player_tiles': player_tiles,
        'wall_tiles': state['wall_tiles'],
        'reward_tiles': state['reward_tiles'],
        'terminal_tiles': state['terminal_tiles'],
        'reward': reward.unsqueeze(-1),
        'terminated': terminated.unsqueeze(-1)
    }
    return TensorDict(out, state.shape)


def _reset(self, tensordict=None):
    batch_size = tensordict.shape if tensordict is not None else self.batch_size
    if tensordict is None or tensordict.is_empty():
        tensordict = self.gen_params(batch_size).to(self.device)
    if '_reset' in tensordict.keys():
        reset_state = self.gen_params(batch_size).to(self.device)
        reset_mask = tensordict['_reset'].squeeze(-1)
        for key in reset_state.keys():
            tensordict[key][reset_mask] = reset_state[key][reset_mask]
    return tensordict


def gen_params(batch_size=None):
    walls = tensor([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ], dtype=torch.float32)

    rewards = tensor([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, -1, 1, 1, 0],
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


def _make_spec(self, td_params):
    batch_size = td_params.shape
    self.observation_spec = CompositeSpec(
        player_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size((*batch_size, 5, 5)),
            dtype=torch.float32,
        ),
        wall_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size((*batch_size, 5, 5)),
            dtype=torch.float32,
        ),
        reward_tiles=UnboundedContinuousTensorSpec(
            shape=torch.Size((*batch_size, 5, 5)),
            dtype=torch.float32,
        ),
        terminal_tiles=BoundedTensorSpec(
            minimum=0,
            maximum=1,
            shape=torch.Size((*batch_size, 5, 5)),
            dtype=torch.bool,
        ),
        player_pos=UnboundedDiscreteTensorSpec(
            shape=torch.Size((*batch_size, 2,)),
            dtype=torch.int64
        ),
        shape=torch.Size((*batch_size,))
    )
    self.state_spec = self.observation_spec.clone()
    self.action_spec = DiscreteTensorSpec(4, shape=torch.Size((*batch_size, 1)))
    self.reward_spec = UnboundedContinuousTensorSpec(shape=torch.Size((*batch_size, 1)))


def _set_seed(self, seed: Optional[int]):
    rng = torch.manual_seed(seed)
    self.rng = rng


class Gridworld(EnvBase):
    metadata = {
        "render_modes": ["human", ""],
        "render_fps": 30
    }
    batch_locked = False

    def __init__(self, td_params=None, device="cpu", batch_size=None):
        if td_params is None:
            td_params = self.gen_params(batch_size)
        super().__init__(device=device, batch_size=batch_size)
        self._make_spec(td_params)

    gen_params = staticmethod(gen_params)
    _make_spec = _make_spec
    _reset = _reset
    _step = staticmethod(_step)
    _set_seed = _set_seed


class RGBFullObsTransform(ObservationTransform):

    def __init__(
            self,
            w: int,
            h: int | None = None,
            in_keys: Sequence[NestedKey] | None = None,
            out_keys: Sequence[NestedKey] | None = None,
    ):
        super().__init__(in_keys, out_keys)
        self.w = w
        self.h = h if h is not None else w

    def forward(self, tensordict):
        return self._call(tensordict)

    # The transform must also modify the data at reset time
    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def _call(self, td):

        player_tiles = td['player_tiles']
        walls = td['wall_tiles']
        rewards = td['reward_tiles']
        terminal = td['terminal_tiles']

        grid = TensorDict({'image': torch.zeros(*walls.shape, 3, dtype=torch.uint8)}, batch_size=td.batch_size)
        grid['image'][walls == 1] = light_gray
        grid['image'][rewards > 0] = green
        grid['image'][rewards < 0] = red
        grid['image'][terminal == 1] = blue
        grid['image'][player_tiles == 1] = yellow
        grid['image'] = grid['image'].permute(0, 3, 1, 2)
        td['observation'] = grid['image'].squeeze(0)
        return td

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec):
        return BoundedTensorSpec(
            minimum=0,
            maximum=255,
            shape=torch.Size((*self.parent.batch_size, 3, self.w, self.h)),
            dtype=torch.uint8,
            device=observation_spec.device
        )


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    import matplotlib.animation as animation
    from torchrl.collectors import SyncDataCollector

    env_batch_size = 64

    env = Gridworld(batch_size=torch.Size([env_batch_size]), device='cuda')
    check_env_specs(env)

    env = TransformedEnv(
        env,
        RGBFullObsTransform(5, in_keys=['wall_tiles'], out_keys=['observation']),
    )

    check_env_specs(env)

    env.append_transform(RewardSum())

    # env.append_transform(Resize(64, 64, in_keys=['observation'], out_keys=['observation'], interpolation='nearest'))
    # check_env_specs(env)

    collector = SyncDataCollector(
        env,
        env.rand_action,
        frames_per_batch=env_batch_size * 1,
        total_frames=env_batch_size * 100,
        split_trajs=False,
        device="cuda",
    )

    buffer = []

    for i, data in enumerate(collector):

        # data = env.rollout(max_steps=100, policy=env.rand_action, break_when_any_terminal=False)
        buffer += [data.cpu()]
        print([data['next', 'reward']])
        print([data['episode_reward'].max().item()])


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

    observation = torch.concatenate([b['observation'] for b in buffer], dim=1)

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