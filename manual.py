import sys
import matplotlib.pyplot as plt
from gridworld import Gridworld, Actions, RGBFullObsTransform
import torch
from torchrl.envs import TransformedEnv, check_env_specs, step_mdp
from torchvision.utils import make_grid

keymap = {"up": Actions.N, "right": Actions.E, "down": Actions.S, "left": Actions.W}

env = Gridworld(batch_size=torch.Size([2]))
env = TransformedEnv(
    env
)
env.append_transform(RGBFullObsTransform())
check_env_specs(env)

global td
td = env.reset()


def on_press(event):
    global td
    print('press', event.key, keymap[event.key])
    sys.stdout.flush()

    td['action'] = torch.tensor([keymap[event.key], keymap[event.key]])
    td = env.step(td)
    td = step_mdp(td)
    img_plt.set_data(td['image'][0].permute(1, 2, 0))
    fig.canvas.draw()

    if td['terminated'][0]:
        plt.pause(0.5)
        td = env.reset()
        img_plt.set_data(td['image'][0].permute(1, 2, 0))
        fig.canvas.draw()


fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)
observation = td['image']
img_plt = ax.imshow(make_grid(observation[0]).permute(1, 2, 0))
ax.set_title('SUPERGRIDWORLD')
plt.show()
