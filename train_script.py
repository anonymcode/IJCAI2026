#!/usr/bin/env python3
"""
Script for training SASRec with command line arguments
Based on my_test_new_era.ipynb
"""

import argparse
import numpy as np
import torch 
import os
import json
from experiment_utils import train_and_eval
from experiment_utils import *

def seed_everything(seed=42):
    """Fix all random seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_default_config():
    """Returns default configuration"""
    return { # config for beer_advocate CE mode from ScalableSASRec
        'batch_size': 32, # for yelp 128 is too big
        'num_blocks': 1, 
        'hidden_units': 64, 
        'num_heads': 2, 
        'lr': 0.001,
        'dropout_rate': 0.3, 
        'l2_emb': 0.1, 
        'sampler_seed': 99, 
        'manual_seed': 37, 
        'num_epochs': 300, 
        'maxlen': 200,
        'use_pos_emb': False, 
        'use_custom_A': False, 
        'mode': 'CE',
        'evaluate_on_every': 3,
        'early_stop_decline_streak': 10,
        'test_eval_buckets': False,
    }
    # return { # default config
    #     'batch_size': 128,
    #     'num_blocks': 2, 
    #     'hidden_units': 64, # dimention of the embedding vector of items and pos_emb
    #     'num_heads': 1, 
    #     'lr': 0.001,
    #     'dropout_rate': 0.2, 
    #     'l2_emb': 0.0, 
    #     'sampler_seed': 99, 
    #     'manual_seed': 111, 
    #     'num_epochs': 300, 
    #     'maxlen': 200,
    #     'use_pos_emb': False, 
    #     'use_custom_A': False, 
    #     'mode': 'CE',
    #     'evaluate_on_every': 10,
    #     'early_stop_decline_streak': 10,
    # }

def main():
    config = get_default_config()

    parser = argparse.ArgumentParser(description='Training SASRec model')
    
    # Main arguments
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Dataset name (e.g., ml-1m, beauty, sports)')
    
    parser.add_argument('--gpu', '-g', type=int, default=config.get('gpu'),
                       help='GPU number to use (default: 0)')
    
    parser.add_argument('--name', '-n', type=str, default='experiment_nemo',
                       help='Experiment name (default: experiment)')
    
    # Model parameters (optional)
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size'),
                       help='Batch size (default: 128)')
    
    parser.add_argument('--num_blocks', type=int, default=config.get('num_blocks'),
                       help='Number of transformer blocks (default: 2)')
    
    parser.add_argument('--hidden_units', type=int, default=config.get('hidden_units'),
                       help='Hidden layer size (default: 64)')
    
    parser.add_argument('--num_heads', type=int, default=config.get('num_heads'),
                       help='Number of attention heads (default: 1)')
    
    parser.add_argument('--lr', type=float, default=config.get('lr'),
                       help='Learning rate (default: 0.001)')
    
    parser.add_argument('--dropout_rate', type=float, default=config.get('dropout_rate'),
                       help='Dropout rate (default: 0.2)')
    
    parser.add_argument('--num_epochs', type=int, default=config.get('num_epochs'),
                       help='Number of epochs (default: 30)')
    
    parser.add_argument('--maxlen', type=int, default=config.get('maxlen'),
                       help='Maximum sequence length (default: 200)')
    
    parser.add_argument('--manual_seed', type=int, default=config.get('manual_seed'),
                       help='Random seed (default: 111)')
    
    parser.add_argument('--sampler_seed', type=int, default=config.get('sampler_seed'),
                       help='Sampler seed (default: 99)')
    
    # Training parameters
    parser.add_argument('--evaluate_on_every', type=int, default=config.get('evaluate_on_every'),
                       help='Evaluate model every N epochs (default: 4)')
    
    parser.add_argument('--early_stop_decline_streak', type=int, default=config.get('early_stop_decline_streak'),
                       help='Number of epochs with decline for early stopping (default: 100)')
    
    # Boolean flags
    parser.add_argument('--use_pos_emb', action='store_true', default=config.get('use_pos_emb'),
                       help='Use positional embeddings')
    
    parser.add_argument('--use_custom_A', action='store_true', default=config.get('use_custom_A'),
                       help='Do NOT use custom attention (default: used)')
    
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Verbose output')
    
    # Save parameters
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save results (default: output)')
    
    parser.add_argument('--save_model', action='store_true',
                       help='Save trained model')
    
    # parser.add_argument('--use_wandb', action='store_true', default=True,
    #                    help='Use wandb for logging (default: disabled)')
    
    parser.add_argument('--no_wandb', action='store_true', default=False,
                       help='Disable wandb logging (takes priority over --use_wandb)')
    parser.add_argument('--use_custom_A_layernorm', action='store_true', default=False)
    
    # KimiDeltaAttention parameters
    parser.add_argument('--use_kimi_attention', action='store_true', default=False,
                       help='Use KimiDeltaAttention (linear attention from Kimi model)')
    parser.add_argument('--kimi_conv_kernel_size', type=int, default=4,
                       help='Convolution kernel size for KimiDeltaAttention (default: 4)')
    
    # ALL-APE parameters
    parser.add_argument('--use_pos_emb_all_layers', action='store_true', default=False,
                       help='ALL-APE: positional embeddings in each attention block (individual)')
    parser.add_argument('--use_pos_emb_all_layers_shared', action='store_true', default=False,
                       help='ALL-APE-shared: one shared positional embedding for all blocks')
    
    # CAPE parameters
    parser.add_argument('--use_cape', action='store_true', default=False,
                       help='CAPE: Context-aware Position Encoding')
    
    # RoPE parameters
    parser.add_argument('--use_rope', action='store_true', default=False,
                       help='RoPE: Rotary Position Embedding')
    
    args = parser.parse_args()
    
    
    # Create configuration
    # config = get_default_config()
    
    # Update configuration from arguments
    config.update({
        'batch_size': args.batch_size,
        'num_blocks': args.num_blocks,
        'hidden_units': args.hidden_units,
        'num_heads': args.num_heads,
        'lr': args.lr,
        'dropout_rate': args.dropout_rate,
        'num_epochs': args.num_epochs,
        'maxlen': args.maxlen,
        'manual_seed': args.manual_seed,
        'sampler_seed': args.sampler_seed,
        'use_pos_emb': args.use_pos_emb,
        'use_custom_A': args.use_custom_A,
        'use_pos_emb_all_layers': args.use_pos_emb_all_layers,
        'use_pos_emb_all_layers_shared': args.use_pos_emb_all_layers_shared,
        'use_cape': args.use_cape,
        'use_rope': args.use_rope,
        'use_wandb': not args.no_wandb,
        'evaluate_on_every': args.evaluate_on_every,
        'early_stop_decline_streak': args.early_stop_decline_streak,
        'use_custom_A_layernorm': args.use_custom_A_layernorm,
        # KimiDeltaAttention config
        'use_kimi_attention': args.use_kimi_attention,
        'kimi_attention_config': {
            'short_conv_kernel_size': args.kimi_conv_kernel_size,
            # head_dim will be computed automatically as hidden_units // num_heads
        },
        'rms_norm_eps': 1e-6,
    })
    
    # Determine model type
    if args.use_kimi_attention:
        # KimiDeltaAttention mode
        if args.use_pos_emb or args.use_custom_A or args.use_pos_emb_all_layers or args.use_pos_emb_all_layers_shared:
            print("WARNING: use_kimi_attention is incompatible with other positional embedding modes")
            print("Disabling all other modes for kimi")
            config['use_pos_emb'] = False
            config['use_custom_A'] = False
            config['use_pos_emb_all_layers'] = False
            config['use_pos_emb_all_layers_shared'] = False
        type_of_model = 'kimi'
    elif args.use_pos_emb_all_layers:
        # ALL-APE mode: positional embeddings in each attention block (individual)
        if args.use_pos_emb or args.use_custom_A or args.use_pos_emb_all_layers_shared:
            print("WARNING: use_pos_emb_all_layers is incompatible with other modes")
            print("Disabling other modes for ALL-APE")
            config['use_pos_emb'] = False
            config['use_custom_A'] = False
            config['use_pos_emb_all_layers_shared'] = False
        type_of_model = 'ALL-APE'
    elif args.use_pos_emb_all_layers_shared:
        # ALL-APE-shared mode: one shared positional embedding for all blocks
        if args.use_pos_emb or args.use_custom_A:
            print("WARNING: use_pos_emb_all_layers_shared is incompatible with other modes")
            print("Disabling other modes for ALL-APE-shared")
            config['use_pos_emb'] = False
            config['use_custom_A'] = False
        type_of_model = 'ALL-APE-shared'
    elif args.use_cape:
        # CAPE mode: Context-aware Position Encoding
        if args.use_pos_emb or args.use_custom_A or args.use_pos_emb_all_layers or args.use_pos_emb_all_layers_shared:
            print("WARNING: use_cape is incompatible with other positional embedding modes")
            print("Disabling other modes for CAPE")
            config['use_pos_emb'] = False
            config['use_custom_A'] = False
            config['use_pos_emb_all_layers'] = False
            config['use_pos_emb_all_layers_shared'] = False
        type_of_model = 'CAPE'
    elif args.use_rope:
        # RoPE mode: Rotary Position Embedding
        if args.use_pos_emb or args.use_custom_A or args.use_pos_emb_all_layers or args.use_pos_emb_all_layers_shared:
            print("WARNING: use_rope is incompatible with other positional embedding modes")
            print("Disabling other modes for RoPE")
            config['use_pos_emb'] = False
            config['use_custom_A'] = False
            config['use_pos_emb_all_layers'] = False
            config['use_pos_emb_all_layers_shared'] = False
        type_of_model = 'RoPE'
    elif args.use_pos_emb==True and args.use_custom_A==True:
        print("ALARM using pos_emb and custom_A")
        exit(1)
    elif args.use_pos_emb==False and args.use_custom_A==False:
        print("ALARM without pos_emb and without custom_A")
        type_of_model = 'empty'
        # print("use_pos_emb and use_custom_A cannot be both False")
        # exit(1)
    else:
        type_of_model = 'custom' if args.use_custom_A else 'normal'
    
    # Device setup
    device = f'cuda:{args.gpu}'
    
    print("=" * 60)
    print(f"🚀 STARTING EXPERIMENT: {args.name}")
    print("=" * 60)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🎮 GPU: {device}")
    print(f"🌱 Random seed: {config['manual_seed']}")
    print(f"📝 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print("=" * 60)
    
    # Fix seed RIGHT BEFORE training for maximum reproducibility
    print(f"🔒 Fixing random seeds (manual_seed={config['manual_seed']})...")
    seed_everything(config['manual_seed'])
    
    try:
        # Start training
        print("starting training...")
        training_results, scores, errors, model = train_and_eval(
            dataset_name=args.dataset,
            topn=10,
            param_config=config,
            device=device,
            verbose=False,
            experiment_name=args.name + '_' + type_of_model + '_' + args.dataset,
            use_wandb=not args.no_wandb,  # Explicitly pass wandb setting

        )
        print("training completed")
        # print("training_results")
        # print(training_results.keys())
        # print(training_results)
        # print("scores")
        # print(scores)
        # print("errors")
        # print(errors.keys())
        
        # Print final results
        # print("\n" + "=" * 60)
        # print("🎉 EXPERIMENT RESULTS")
        # print("=" * 60)
        # print(f"📈 Best epoch: {training_results.get('best_epoch', 'N/A')}")
        # print(f"📊 Test metrics:")
        # for metric, value in scores.items():
        #     error = errors.get(metric, 0)
        #     if not np.isnan(error):
        #         print(f"   {metric}: {value:.4f} ± {error:.4f}")
        #     else:
        #         print(f"   {metric}: {value:.4f}")
        # print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 
