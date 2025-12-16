# LLM Initialization Transfer Experiment

This project implements an experiment to test whether parameter distribution statistics from a larger trained model can improve the training convergence of a smaller model in the same architecture family.

## Features

- **Statistics Extraction**: Extract parameter statistics from pre-trained models
- **Smart Initialization**: Initialize smaller models with transferred Gaussian statistics
- **Efficient Training**: Uses TRL's SFTTrainer with packing for optimal performance
- **Comprehensive Analysis**: Automated comparison and visualization of training results

## Project Structure

- `extract_stats.py`: Extract parameter statistics from a pre-trained source model
- `initialize_model.py`: Initialize target model with transferred statistics
- `train_comparison.py`: Train both models and compare convergence using SFTTrainer
- `analyze_results.py`: Analyze and visualize training results
- `config/`: Configuration files
- `utils/`: Utility modules for data, model, and logging

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the experiment:
   ```bash
   # Phase 1: Extract statistics
   python extract_stats.py --model_name "Qwen/Qwen2.5-1.5B" --output_path "./stats/qwen_1.5b_stats.json"

   # Phase 2: Initialize models
   python initialize_model.py --stats_path "./stats/qwen_1.5b_stats.json" --target_config "./config/default_config.yaml" --output_dir "./models/initialized" --temperature 1.0

   # Phase 3: Train with comparison
   python train_comparison.py --model_dir "./models/initialized" --config_path "./config/default_config.yaml" --dataset "wikitext-2" --output_dir "./experiments/run_001"

   # Phase 4: Analyze results
   python analyze_results.py --experiment_dir "./experiments/run_001" --output_dir "./results/run_001"
   ```

## Key Technologies

- **TRL SFTTrainer**: For efficient supervised fine-tuning with automatic dataset processing and packing
- **HuggingFace Transformers**: Model loading and configuration
- **PyTorch**: Deep learning framework
- **Matplotlib/Seaborn**: Visualization and analysis

## Requirements

- Python 3.8+
- PyTorch 2.0+
- HuggingFace Transformers 4.30+