# Models Layout

`models/` only stores model-specific implementations.

Current convention:

- `models/memflow/`: current usable backend, containing memflow-specific training, inference, pipelines, trainer and WAN code.
- `models/self_forcing/`: reserved slot for future self-forcing implementation.
- `models/causal_forcing/`: reserved slot for future causal-forcing implementation.

Shared layers stay at repository root:

- `rewards/`: reward functions and reward-model assets shared by all models.
- `utils/`: generic utilities.
- `configs/`: configuration files.
- top-level `train*.py` / `inference*.py`: compatibility entrypoints that forward to the concrete model implementation.

Recommended rule when adding a new model:

1. Put model-specific code under its own subdirectory.
2. Keep reward composition under `rewards/`.
3. Give the model its own training script, for example `train_rl.py`.
4. Only keep thin compatibility wrappers at repository root when needed.

