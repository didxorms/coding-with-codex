# CartPole DQN (CPU) with PyTorch

This repository contains a minimal Deep Q-Network (DQN) implementation for the
CartPole environment, designed to run on CPU and log training metrics to CSV
with a simple reward plot.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --episodes 300 --save-best --record-gif
```

Training outputs:

- `runs/<timestamp>/metrics.csv`: per-episode rewards.
- `runs/<timestamp>/reward_plot.png`: reward curve.
- `runs/<timestamp>/checkpoint.pt`: trained network weights.
- `runs/<timestamp>/best_checkpoint.pt`: best checkpoint by 10-episode average reward.
- `runs/<timestamp>/best_agent.gif`: animation from the best checkpoint.

## Project structure

```
.
├── README.md
├── requirements.txt
├── src
│   ├── model.py
│   ├── replay_buffer.py
│   ├── train.py
│   └── utils.py
└── .gitignore
```

## Notes

- Uses `gymnasium` and the classic control `CartPole-v1` environment.
- CPU-only by default; no CUDA required.

## Hyperparameter tuning tips

Try adjusting these flags for more stable learning:

- `--epsilon-decay 500` to keep exploration longer.
- `--target-update 5` to refresh the target network more often.
- `--batch-size 128` to smooth gradients.
