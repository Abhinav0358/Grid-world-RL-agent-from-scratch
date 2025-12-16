from Grid_env import gridenv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

i = 0
num_episodes = 1000
env = gridenv()
epsilon = 0.7
alpha = 0.05  # Learning rate
gamma = 0.99  # Discount factor
total_steps=[]

# Where to save per-episode path plots
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# Episodes to visualize (only those <= num_episodes will be collected)
checkpoint_episodes = [10, 20, 30, 50, 100, 200, 500, 1000]
episode_paths = {}

while(i<num_episodes):
    state = env.reset()
    # print(state)
    done = False
    steps = 0
    path = [tuple(state)]  # start position
    while(not done):
        prev_state = tuple(state)  # freeze before env.step mutates
        valid_action_space = env.action_space
        if prev_state[0] == 0: # left edge
            valid_action_space = [a for a in valid_action_space if a != 0]
        if prev_state[0] == 3: # right edge
            valid_action_space = [a for a in valid_action_space if a != 1]
        if prev_state[1] == 0: # bottom edge
            valid_action_space = [a for a in valid_action_space if a != 2]
        if prev_state[1] == 3: # top edge
            valid_action_space = [a for a in valid_action_space if a != 3]

        probability = np.random.rand()
        if(probability < epsilon):
            action = np.random.choice(valid_action_space)
        else:
            max_q_value = float('-inf')
            for a in valid_action_space:
                q_value = env.q_values[(prev_state, a)]
                if q_value > max_q_value:
                    max_q_value = q_value
                    action = a
        next_state, reward, terminated, truncated = env.step(action)
        path.append(tuple(next_state))
        # if(i<5 or i>995):
        #     print(f"Episode: {i+1}, State: {prev_state}, valid action space{[a for a in valid_action_space]} Action: {action}, Reward: {reward}, Next State: {next_state}")
        steps+=1
        current_q_value = env.q_values[(prev_state, action)]
        max_future_q_value = max([env.q_values[(tuple(next_state), a)] for a in env.action_space])
        env.q_values[(prev_state, action)] = current_q_value + alpha * (reward + gamma * max_future_q_value - current_q_value)
        state = list(next_state)
        done = terminated or truncated
        epsilon = max(0.05, epsilon * 0.995)  # Decay epsilon
    total_steps.append(steps)
    episode_num = i + 1
    if episode_num in checkpoint_episodes:
        episode_paths[episode_num] = path
    i += 1

# print number of steps for each episode
for episode_idx, episode_step in enumerate(total_steps, start=1):
    print(f"total steps in episode {episode_idx}: {episode_step}")


def plot_paths(paths, grid_size=(4, 4)):
    if not paths:
        print("No paths to plot (none of the checkpoints occurred within num_episodes).")
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    width, height = grid_size
    ax.set_xticks(range(width+1))
    ax.set_yticks(range(height+1))
    ax.grid(True)
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(-0.5, height-0.5)
    ax.invert_yaxis()  # keep (0,0) visually at bottom-left
    ax.scatter(3, 3, c='gold', marker='*', s=200, label='Goal (3,3)')
    colors = plt.cm.viridis(np.linspace(0, 1, len(paths)))
    for (ep, coords), color in zip(sorted(paths.items()), colors):
        if len(coords) < 2:
            continue
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        ax.plot(xs, ys, '-o', color=color, label=f'Episode {ep}')
        ax.scatter(xs[0], ys[0], c=color, marker='s', s=80)  # start
    ax.set_title('Episode paths at checkpoints')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_and_save_path(coords, episode_num, grid_size=(4, 4), output_dir=OUTPUT_DIR):
    if len(coords) < 2:
        print(f"Episode {episode_num}: not enough points to plot.")
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    width, height = grid_size
    ax.set_xticks(range(width+1))
    ax.set_yticks(range(height+1))
    ax.grid(True)
    ax.set_xlim(-0.5, width-0.5)
    ax.set_ylim(-0.5, height-0.5)
    ax.invert_yaxis()
    ax.scatter(3, 3, c='gold', marker='*', s=180, label='Goal (3,3)')
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    ax.plot(xs, ys, '-o', color='steelblue', label=f'Episode {episode_num}')
    ax.scatter(xs[0], ys[0], c='steelblue', marker='s', s=80)  # start
    ax.set_title(f'Episode {episode_num} path')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    outfile = output_dir / f"episode_{episode_num}.png"
    plt.savefig(outfile)
    plt.close(fig)
    print(f"Saved path plot for episode {episode_num} to {outfile}")


plot_paths({ep: episode_paths[ep] for ep in episode_paths})

# Save per-episode plots for checkpoints
for ep, coords in episode_paths.items():
    plot_and_save_path(coords, ep)