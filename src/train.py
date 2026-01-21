from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from model import DQN
from replay_buffer import ReplayBuffer, Transition
from utils import plot_rewards, set_seed, write_metrics_csv


def select_action(
    q_net: DQN,
    state: np.ndarray,
    epsilon: float,
    num_actions: int,
    device: torch.device,
) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = q_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def compute_epsilon(
    episode: int,
    start: float,
    end: float,
    decay: float,
) -> float:
    return end + (start - end) * np.exp(-1.0 * episode / decay)


def optimize(
    q_net: DQN,
    target_net: DQN,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    if len(buffer) < batch_size:
        return 0.0

    transitions = buffer.sample(batch_size)
    states = torch.tensor(
        np.array([t.state for t in transitions]),
        dtype=torch.float32,
        device=device,
    )
    actions = torch.tensor(
        [t.action for t in transitions],
        dtype=torch.int64,
        device=device,
    ).unsqueeze(1)
    rewards = torch.tensor(
        [t.reward for t in transitions],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    next_states = torch.tensor(
        np.array([t.next_state for t in transitions]),
        dtype=torch.float32,
        device=device,
    )
    dones = torch.tensor(
        [t.done for t in transitions],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)

    q_values = q_net(states).gather(1, actions)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1, keepdim=True)[0]
        target = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.functional.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN for CartPole-v1 (CPU)")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cpu")

    env = gym.make("CartPole-v1")
    env.action_space.seed(args.seed)

    obs_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    q_net = DQN(obs_size, num_actions).to(device)
    target_net = DQN(obs_size, num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay_buffer = ReplayBuffer(args.buffer_size)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or Path("runs") / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    rewards_history: Deque[float] = deque(maxlen=args.episodes)
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        epsilon = compute_epsilon(
            episode,
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay,
        )

        for _ in range(args.max_steps):
            action = select_action(q_net, state, epsilon, num_actions, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(
                Transition(
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                    done=done,
                )
            )
            episode_reward += reward
            state = next_state

            optimize(
                q_net,
                target_net,
                optimizer,
                replay_buffer,
                args.batch_size,
                args.gamma,
                device,
            )

            if done:
                break

        rewards_history.append(float(episode_reward))

        if episode % args.target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        avg_reward = np.mean(list(rewards_history)[-10:])
        print(
            f"Episode {episode:4d} | "
            f"Reward: {episode_reward:6.1f} | "
            f"Avg(10): {avg_reward:6.1f} | "
            f"Epsilon: {epsilon:.3f}"
        )

    env.close()

    metrics_path = run_dir / "metrics.csv"
    write_metrics_csv(metrics_path, rewards_history)
    plot_rewards(run_dir / "reward_plot.png", rewards_history)
    torch.save(q_net.state_dict(), run_dir / "checkpoint.pt")
    print(f"Saved outputs to {run_dir}")


if __name__ == "__main__":
    main()
