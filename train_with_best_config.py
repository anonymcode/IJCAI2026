#!/usr/bin/env python3
"""
Script for training SASRec using the BEST configuration from Optuna.
Loads `best_config.json` from the folder corresponding to the dataset and model type.
Extended version of `train_with_best_config.py` with support for additional model types.

Usage example:

$ python train_with_best_config_all.py --dataset ml-1m --type_of_model normal --gpu 0
"""

import argparse
import json
import os
import sys
import numpy as np
import torch

from experiment_utils import train_and_eval
from experiment_utils import *  # noqa: F403,F401 (for legacy util functions)
import defaults

# ---------- HELPER FUNCTIONS ---------- #

def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_best_config(dataset: str, model_name: str, logs_dir: str = 'optuna_logs', study_name: str = '') -> dict:
    """Loads the best configuration from a JSON file for a specific dataset."""
    config_path = os.path.join(logs_dir, f'{study_name}_{model_name}_{dataset}', 'best_config.json')
    if not os.path.exists(config_path):
        print(f"❌ Error: Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"✅ Configuration successfully loaded from {config_path}")
    return config


# ---------- MAIN SCRIPT LOGIC ---------- #

def train_dataset(
    dataset: str,
    device: str,
    type_of_model: str,
    type_of_custom_A: int,
    type_of_trinagularity: str,
    type_of_connection: int,
    study_name: str,
    name: str,
    num_epochs: int,
    manual_seed: int,
    sampler_seed: int,
    evaluate_on_every: int,
    val_eval_mode: str,
    test_eval_mode: str,
    early_stop_decline_streak: int,
    no_wandb: bool,
    use_custom_A_layernorm: bool,
    use_confidence_intervals: bool,
) -> bool:
    """Trains the model on a single dataset."""
    print(f"\n{'=' * 60}")
    print(f"📊 Starting training on dataset: {dataset}")
    print(f"{'=' * 60}")

    # Load config
    model_name = f"{str(type_of_model)}_[{type_of_custom_A}-{type_of_trinagularity}-{type_of_connection}]"
    config = load_best_config(dataset, model_name, defaults.optuna_dir, study_name=study_name)

    # Update config with global CLI parameters
    config.update({
        'num_epochs': num_epochs,
        'manual_seed': manual_seed,
        'sampler_seed': sampler_seed,
        'evaluate_on_every': evaluate_on_every,
        
        'val_eval_mode': val_eval_mode,
        'test_eval_mode': test_eval_mode,

        'early_stop_decline_streak': early_stop_decline_streak,
        'use_wandb': not no_wandb,
        'mode': 'CE',
        'test_eval_buckets': False,
        'use_custom_A_layernorm': use_custom_A_layernorm,
        'use_confidence_intervals': use_confidence_intervals,
        'save_model': True,
        'study_name': study_name,
    })

    # Model flags
    if type_of_model == 'empty':
        config['use_pos_emb'] = False
        config['use_custom_A'] = False
        config['use_kimi_attention'] = False
    elif type_of_model == 'normal':
        config['use_pos_emb'] = True
        config['use_custom_A'] = False
        config['use_kimi_attention'] = False
    elif type_of_model == 'kimi':
        config['use_pos_emb'] = False
        config['use_custom_A'] = False
        config['use_kimi_attention'] = True
        # Default settings for KimiDeltaAttention
        if 'kimi_attention_config' not in config:
            config['kimi_attention_config'] = {
                'short_conv_kernel_size': 4,
                'head_dim': config['hidden_units'] // config['num_heads'],
            }
        if 'kimi_mode' not in config:
            config['kimi_mode'] = 'chunk'
        if 'rms_norm_eps' not in config:
            config['rms_norm_eps'] = 1e-6
    elif type_of_model == 'PAA':
        config['use_custom_A'] = True
        config['use_pos_emb'] = False
        config['use_custom_A_layernorm'] = False
        config['type_of_custom_A'] = type_of_custom_A
        config['type_of_trinagularity'] = type_of_trinagularity
        config['A&A-T'] = type_of_connection
    elif type_of_model == 'ALL-APE':
        # ALL-APE: positional embeddings in each attention block (individual)
        config['use_pos_emb'] = False
        config['use_pos_emb_all_layers'] = True
        config['use_pos_emb_all_layers_shared'] = False
        config['use_custom_A'] = False
        config['use_kimi_attention'] = False
    elif type_of_model == 'ALL-APE-shared':
        # ALL-APE-shared: one shared positional embedding for all blocks
        config['use_pos_emb'] = False
        config['use_pos_emb_all_layers'] = False
        config['use_pos_emb_all_layers_shared'] = True
        config['use_custom_A'] = False
        config['use_kimi_attention'] = False
    elif type_of_model == 'CAPE':
        # CAPE: Context-aware Position Encoding
        config['use_pos_emb'] = False
        config['use_pos_emb_all_layers'] = False
        config['use_pos_emb_all_layers_shared'] = False
        config['use_cape'] = True
        config['use_rope'] = False
        config['use_custom_A'] = False
        config['use_kimi_attention'] = False
    elif type_of_model == 'RoPE':
        # RoPE: Rotary Position Embedding
        config['use_pos_emb'] = False
        config['use_pos_emb_all_layers'] = False
        config['use_pos_emb_all_layers_shared'] = False
        config['use_cape'] = False
        config['use_rope'] = True
        config['use_custom_A'] = False
        config['use_kimi_attention'] = False
    else: 
        print(f"❌ Unknown model type: {type_of_model}")
        return False

    # Seed before training
    seed_everything(config['manual_seed'])

    experiment_name = f"{name}_{type_of_model}_[{type_of_custom_A}-{type_of_trinagularity}-{type_of_connection}]_{dataset}"

    try:
        print(f"🚀 Starting training on device: {device}")
        training_results, scores, errors, model = train_and_eval(
            dataset_name=dataset,
            topn=10,
            param_config=config,
            device=device,
            verbose=False,
            experiment_name=experiment_name,
            use_wandb=not no_wandb,
        )
        print(f"✅ {dataset} completed. Metrics: {scores if scores else 'N/A'}")
        return True
    except Exception as e:
        print(f"❌ Error on {dataset}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Training SASRec model with best configuration'
    )

    # Main arguments
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='Dataset name (e.g., ml-1m, beauty, sports)')

    parser.add_argument('--type_of_model', '-t', type=str, required=True,
                        help='Model type')

    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU number to use (default: 0)')

    parser.add_argument('--name', '-n', type=str, default='best_config_experiment_multi',
                        help='Base experiment name')

    # Parameters that may be missing in config
    parser.add_argument('--num_epochs', type=int, default=300,
                        help='Number of epochs')
    parser.add_argument('--manual_seed', type=int, default=111,
                        help='Random seed')
    parser.add_argument('--sampler_seed', type=int, default=99,
                        help='Sampler seed')
    parser.add_argument('--evaluate_on_every', type=int, default=1,
                        help='Evaluate model every N epochs')
    parser.add_argument('--early_stop_decline_streak', type=int, default=20,
                        help='Early stopping threshold')
    parser.add_argument('--study_name', type=str, default='exclude',
                        help='Study name')

    parser.add_argument('--val_eval_mode', type=str, default='successive',
                        help='Validation evaluation mode') 
    parser.add_argument('--test_eval_mode', type=str, default='successive',
                        help='Test evaluation mode') 

    parser.add_argument('--no_wandb', action='store_true', default=False,
                        help='Disable wandb logging')
    parser.add_argument('--use_custom_A_layernorm', action='store_true', default=False)
    parser.add_argument('--upper_triangular', action='store_true', default=False)
    
    # Confidence intervals flag for final test evaluation
    parser.add_argument('--use_confidence_intervals', action='store_true', default=False,
                        help='Enable test evaluation with confidence intervals (evaluate_w_conf_intervals)')
    # Matrix A type
    parser.add_argument('--type_of_custom_A', type=int, default=0)
    parser.add_argument('--type_of_trinagularity', type=str, default="none")
    parser.add_argument('--type_of_connection', type=int, default=0)
    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available() and args.gpu is not None and args.gpu < torch.cuda.device_count():
        device = f'cuda:{args.gpu}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print("=" * 60)
    print(f"🚀 STARTING EXPERIMENT WITH BEST CONFIGURATION: {args.name}")
    print("=" * 60)
    print(f"📊 Dataset: {args.dataset}")
    print(f"⚙️ Model type: {args.type_of_model}")
    print(f"🎮 Device: {device}")
    print(f"🌱 Random seed: {args.manual_seed}")
    print("=" * 60)

    # Train on dataset
    success = train_dataset(
        dataset=args.dataset,
        device=device,
        type_of_model=args.type_of_model,
        type_of_custom_A=args.type_of_custom_A,
        type_of_trinagularity=args.type_of_trinagularity,
        type_of_connection=args.type_of_connection,
        study_name=args.study_name,
        name=args.name,
        num_epochs=args.num_epochs,
        manual_seed=args.manual_seed,
        sampler_seed=args.sampler_seed,
        evaluate_on_every=args.evaluate_on_every,
        val_eval_mode=args.val_eval_mode,
        test_eval_mode=args.test_eval_mode,
        early_stop_decline_streak=args.early_stop_decline_streak,
        no_wandb=args.no_wandb,
        use_custom_A_layernorm=args.use_custom_A_layernorm,
        use_confidence_intervals=args.use_confidence_intervals,
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
