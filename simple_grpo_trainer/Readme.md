# Simple GRPO Trainer

This repository provides a basic implementation of a GRPO (Group Relative Policy Optimization) Trainer for training Large Language Models (LLMs). It includes dataset processing, reward functions, and training logic using PyTorch Lightning and Hugging Face Transformers.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd simple_grpo_trainer
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install uv
   uv sync
   # or, if using pyproject.toml:
   pip install . -e
   ```

## Dependencies

Main dependencies (see `pyproject.toml` for details):
- Python >= 3.12
- datasets
- hf-transfer, hf-xet
- lightning (PyTorch Lightning)
- loguru
- peft
- pytest, ruff
- tensorboard
- transformers
- trl

## Running the Code

- **Training:**
  - The main training logic is in `src/GRPOTrainer.py`. You can use or extend the `SimpleGRPOModule` class for training LLMs with GRPO.
- **Dataset Processing:**
  - Dataset loading and processing utilities are in `src/dataset_processor.py`.
- **Rewards:**
  - Reward functions for training are in `src/rewards.py`.

## Running Tests

To run all tests:
```bash
pytest
```
Or to run a specific test file:
```bash
pytest test/test_dataset_processor.py
```

## Project Structure

- `src/`
  - `GRPOTrainer.py`: Main trainer and model logic for GRPO.
  - `dataset_processor.py`: Functions for loading, processing, and tokenizing datasets.
  - `rewards.py`: Reward calculation utilities for model training.
  - `schemas.py`: Type definitions and schemas.
- `test/`
  - `test_dataset_processor.py`: Unit tests for dataset processing functions.
  - `test_rewards.py`: Unit tests for reward functions.
  - `test_trainer.py`: Unit tests for the trainer logic.
- `pyproject.toml`: Project dependencies and configuration.
- `Readme.md`: This file.

## Notes
- This codebase is designed for research and educational purposes. Extend or adapt as needed for your use case.
- For questions or issues, please open an issue or pull request.