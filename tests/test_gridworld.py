from gridworld import Gridworld, Actions
import torch
from torch import tensor
from torchrl.envs import step_mdp


def test_batch_size_one():
    env = Gridworld(batch_size=1, device='cpu')
    state = env.rollout(10)
    assert state.batch_size[0] == 1


def test_movement_and_walls():
    env = Gridworld(batch_size=torch.Size([4]), device='cpu')
    state = env.reset()
    state['action'] = torch.Tensor([Actions.N, Actions.E, Actions.S, Actions.W]).long().unsqueeze(-1)

    state = env.step(state)
    state = step_mdp(state)

    north = tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=torch.uint8)

    east = tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=torch.uint8)

    south = tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]], dtype=torch.uint8)

    west = tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=torch.uint8)

    assert (state['player_tiles'][0] == north).all()
    assert (state['player_tiles'][1] == east).all()
    assert (state['player_tiles'][2] == south).all()
    assert (state['player_tiles'][3] == west).all()

    state['action'] = torch.Tensor([Actions.N, Actions.E, Actions.S, Actions.W]).long().unsqueeze(-1)
    state = env.step(state)
    state = step_mdp(state)

    assert (state['player_tiles'][0] == north).all()
    assert (state['player_tiles'][1] == east).all()
    assert (state['player_tiles'][2] == south).all()
    assert (state['player_tiles'][3] == west).all()


def test_corners():
    env = Gridworld(batch_size=torch.Size([4]), device='cpu')
    state = env.reset()
    state['action'] = torch.Tensor([Actions.N, Actions.E, Actions.S, Actions.W]).long().unsqueeze(-1)
    state = env.step(state)
    state = step_mdp(state)
    state['action'] = torch.Tensor([Actions.E, Actions.S, Actions.W, Actions.N]).long().unsqueeze(-1)
    state = env.step(state)
    state = step_mdp(state)

    northeast = tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=torch.uint8)

    southeast = tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]], dtype=torch.uint8)

    southwest = tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=torch.uint8)

    northwest = tensor([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]], dtype=torch.uint8)

    reward = tensor([-1., 1., -1, 1]).unsqueeze(-1)
    done = tensor([True, False, True, False]).unsqueeze(-1)

    assert (state['player_tiles'][0] == northeast).all()
    assert (state['player_tiles'][1] == southeast).all()
    assert (state['player_tiles'][2] == southwest).all()
    assert (state['player_tiles'][3] == northwest).all()
    assert (state['done'] == done).all()
    assert (state['reward' == reward]).all()

