import argparse
import numpy as np
import torch 
import os
import json
import optuna
from optuna.samplers import TPESampler
from experiment_utils import train_and_eval
from experiment_utils import *
import defaults

def seed_everything(seed=42):
    """Fix all random seeds"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



def main():
    parser = argparse.ArgumentParser(description='SASRec hyperparameter optimization with Optuna')
    
    # Main arguments
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Dataset name (e.g., ml-1m, beauty, sports)')
    
    parser.add_argument('--gpu', '-g', type=str, default='0',
                       help='GPU number to use (default: 0)')
    
    parser.add_argument('--study_name', '-s', type=str, default='optuna',
                       help='Optuna study name')
    
    parser.add_argument('--n_trials', '-t', type=int, required=True,
                       help='Number of optimization trials (default: 100)')
    
    parser.add_argument('--target_metric', type=str, default='NDCG@10',
                       help='Target metric for optimization')
    
    parser.add_argument('--model_type', type=str, default='custom',
                       help='Model type')
    

    parser.add_argument('--timeout', type=int, default=None,
                       help='Maximum optimization time in seconds')
    
    # boolean arguments
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                       help='Verbose output')
    parser.add_argument('--storage', action='store_true', default=True,
                       help='Save to database (e.g., sqlite:///optuna.db)')
    parser.add_argument('--load_if_exists', action='store_true', default=False,
                       help='Load existing study if it exists')
    parser.add_argument('--yes_wandb', action='store_true', default=False,
                       help='Use Weights and Biases for logging')
    # parser.add_argument('--upper_triangular', action='store_true', default=False)
    parser.add_argument('--cap_total_trials', action='store_true', default=False,
                       help='Interpret n_trials as target TOTAL number of trials in study (resumption will not be additive)')

    parser.add_argument('--type_of_custom_A', type=int, required=True)
    parser.add_argument('--type_of_trinagularity', type=str, required=True)
    parser.add_argument('--type_of_connection', type=int, required=True)
    
    

    
    args = parser.parse_args()
    
    # Device setup
    device = f'cuda:{args.gpu}'
    
    print("=" * 80)
    print(f"🔬 STARTING HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"📊 Dataset: {args.dataset}")
    print(f"🎮 GPU: {device}")
    print(f"🎯 Target metric: {args.target_metric}")
    print(f"🔄 Number of trials: {args.n_trials}")
    print(f"🤖 Model type: {args.model_type}")
    print(f"📝 Study name: {args.study_name}")
    print("=" * 80)
    
    base_config = {
        'sampler_seed': 99, 
        'manual_seed': 37,

        'num_epochs': 300,
        'mode': 'CE',
        'evaluate_on_every': 1,
        'early_stop_decline_streak': 10,

        'val_eval_mode': 'successive',
        'test_eval_mode': 'successive',
    }
    if args.model_type == 'empty':
        base_config['use_custom_A'] = False
        base_config['use_pos_emb'] = False
        base_config['use_kimi_attention'] = False

    elif args.model_type == 'normal':
        base_config['use_custom_A'] = False
        base_config['use_pos_emb'] = True
        base_config['use_kimi_attention'] = False


    elif args.model_type == 'PAA':
        base_config['use_custom_A'] = True
        base_config['use_pos_emb'] = False
        # base_config['use_custom_A_layernorm'] = False

        base_config['type_of_custom_A'] = args.type_of_custom_A
        base_config['type_of_trinagularity'] = args.type_of_trinagularity
        base_config['type_of_connection'] = args.type_of_connection

    elif args.model_type == 'ALL-APE':
        # ALL-APE: positional embeddings in each attention block (individual)
        base_config['use_custom_A'] = False
        base_config['use_pos_emb'] = False
        base_config['use_pos_emb_all_layers'] = True
        base_config['use_pos_emb_all_layers_shared'] = False
        base_config['use_kimi_attention'] = False

    elif args.model_type == 'ALL-APE-shared':
        # ALL-APE-shared: one shared positional embedding for all blocks
        base_config['use_custom_A'] = False
        base_config['use_pos_emb'] = False
        base_config['use_pos_emb_all_layers'] = False
        base_config['use_pos_emb_all_layers_shared'] = True
        base_config['use_kimi_attention'] = False

    elif args.model_type == 'CAPE':
        # CAPE: Context-aware Position Encoding
        base_config['use_custom_A'] = False
        base_config['use_pos_emb'] = False
        base_config['use_pos_emb_all_layers'] = False
        base_config['use_pos_emb_all_layers_shared'] = False
        base_config['use_cape'] = True
        base_config['use_rope'] = False
        base_config['use_kimi_attention'] = False

    elif args.model_type == 'RoPE':
        # RoPE: Rotary Position Embedding
        base_config['use_custom_A'] = False
        base_config['use_pos_emb'] = False
        base_config['use_pos_emb_all_layers'] = False
        base_config['use_pos_emb_all_layers_shared'] = False
        base_config['use_cape'] = False
        base_config['use_rope'] = True
        base_config['use_kimi_attention'] = False

    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    
    if args.storage: 
        storage = f"sqlite:///{defaults.optuna_db_path}"
        print(f"Storage: {storage}")
    else:
        storage = None

    # study_name = f"{args.study_name}_{args.model_type}-{args.type_of_trinagularity}_{args.dataset}"
    # formula type, matrix type, matrix connection type
    study_name = f"{args.study_name}_{args.model_type}_[{args.type_of_custom_A}-{args.type_of_trinagularity}-{args.type_of_connection}]_{args.dataset}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',  # Maximize target metric
        sampler=TPESampler(seed=base_config['manual_seed']),
        storage=storage,
        load_if_exists=args.load_if_exists
    )
    
    # Determine how many trials to run in this run
    if args.cap_total_trials and args.load_if_exists:
        total_existing_trials = len(study.trials)
        remaining_n_trials = max(0, args.n_trials - total_existing_trials)
        print(f"🧮 Total trial cap mode activated: target={args.n_trials}, existing={total_existing_trials}, will run={remaining_n_trials}")
    else:
        remaining_n_trials = args.n_trials
        if args.load_if_exists:
            print(f"🧮 Additive trial mode: existing={len(study.trials)}, will add={remaining_n_trials}")

    # Create objective function with base configuration
    
    def objective_function(trial):
        # Sample hyperparameters
        config = base_config.copy()
        
        # if args.dataset == 'ml-1m':
        #     config.update({  # ml-1m  # 216
        #         'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        #         'num_blocks': trial.suggest_categorical('num_blocks', [1, 2, 3]),
        #         'hidden_units': trial.suggest_categorical('hidden_units', [64, 128, 256]),
        #         'num_heads': trial.suggest_categorical('num_heads', [1]),
        #         'lr': trial.suggest_categorical('lr', [3e-4, 1e-3]),
        #         'dropout_rate': trial.suggest_categorical('dropout_rate', [0.1, 0.3]),
        #         'maxlen': trial.suggest_categorical('maxlen', [128]),
        #         'l2_emb': trial.suggest_categorical('l2_emb', [0.0]),
        #     })
        if args.dataset == 'listens' or args.dataset == 'zvuk':
            config.update({  # 108
                'batch_size': trial.suggest_categorical('batch_size', [32]),
                'num_blocks': trial.suggest_categorical('num_blocks', [1, 2, 3]),
                'hidden_units': trial.suggest_categorical('hidden_units', [64, 128, 256]),
                'num_heads': trial.suggest_categorical('num_heads', [1]),
                'lr': trial.suggest_categorical('lr', [3e-4, 1e-3]),
                'dropout_rate': trial.suggest_categorical('dropout_rate', [0.1, 0.3]),
                'maxlen': trial.suggest_categorical('maxlen', [128]),
                'l2_emb': trial.suggest_categorical('l2_emb', [0.0]),
            })
        else:
            config.update({ # 216
                'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
                'num_blocks': trial.suggest_categorical('num_blocks', [1, 2, 3]),
                'hidden_units': trial.suggest_categorical('hidden_units', [64, 128, 256]),
                'num_heads': trial.suggest_categorical('num_heads', [1]),
                'lr': trial.suggest_categorical('lr', [3e-4, 1e-3]),
                'dropout_rate': trial.suggest_categorical('dropout_rate', [0.1, 0.3]),
                'maxlen': trial.suggest_categorical('maxlen', [128]),
                'l2_emb': trial.suggest_categorical('l2_emb', [0.0]),
            })

        # config.update({ # gowalla
        #     'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        #     'num_blocks': trial.suggest_categorical('num_blocks', [1, 2, 3]),
        #     'hidden_units': trial.suggest_categorical('hidden_units', [64, 128, 256]),
        #     'num_heads': trial.suggest_categorical('num_heads', [1, 2, 4]),
        #     'lr': trial.suggest_categorical('lr', [3e-4, 1e-3]),
        #     'dropout_rate': trial.suggest_categorical('dropout_rate', [0.1, 0.3]),
        #     'maxlen': trial.suggest_categorical('maxlen', [128]),
        #     # 'use_custom_A_layernorm': trial.suggest_categorical('use_custom_A_layernorm', [True, False]), # for custom model
        #     'l2_emb': trial.suggest_categorical('l2_emb', [0.0]),
        # })
        # config.update({ # yelp
        #     'batch_size': trial.suggest_categorical('batch_size', [128, 256]),
        #     'num_blocks': trial.suggest_categorical('num_blocks', [1, 2, 3]),
        #     'hidden_units': trial.suggest_categorical('hidden_units', [64, 128, 256]),
        #     'num_heads': trial.suggest_categorical('num_heads', [1, 2, 4]),
        #     'lr': trial.suggest_categorical('lr', [3e-4, 1e-3]),
        #     'dropout_rate': trial.suggest_categorical('dropout_rate', [0.1, 0.3]),
        #     'maxlen': trial.suggest_categorical('maxlen', [128]),
        #     # 'use_custom_A_layernorm': trial.suggest_categorical('use_custom_A_layernorm', [True, False]), # for custom model
        #     'l2_emb': trial.suggest_categorical('l2_emb', [0.0]),
        # })
        
        seed_everything(config['manual_seed'])
        
        try:
            # Start training
            training_results, scores, errors, model = train_and_eval(
                dataset_name=args.dataset,
                topn=10,
                param_config=config,
                device=device,
                verbose=args.verbose,
                experiment_name=f"{study_name}_trial_{trial.number}",
                use_wandb=args.yes_wandb
            )
            
            # Return target metric
            if args.target_metric not in scores:
                available_metrics = list(scores.keys())
                raise ValueError(f"Target metric '{args.target_metric}' not found in results! "
                               f"Available metrics: {available_metrics}")
            
            target_score = scores[args.target_metric]
            
            # Log additional metrics as trial attributes
            for metric, score in scores.items():
                trial.set_user_attr(f'final_{metric}', score)
            
            # Log configuration
            for key, value in config.items():
                trial.set_user_attr(f'config_{key}', value)
            
            return target_score
            
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {str(e)}")
            return 0.0
    
    try:
        # Start optimization
        print("🚀 Starting optimization...")
        study.optimize(
            objective_function, 
            n_trials=remaining_n_trials,
            timeout=args.timeout,
            catch=(Exception,)  # Continue optimization even on errors
        )
        
        # Print results
        print("\n" + "=" * 80)
        print("🎉 OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"📈 Best {args.target_metric} value: {study.best_value:.4f}")
        print(f"📊 Best parameters:")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")
        
        # Save results in organized structure
        optuna_logs_dir = defaults.optuna_dir
        results_dir = os.path.join(optuna_logs_dir, f"{study_name}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save best parameters
        best_config = base_config.copy()
        best_config.update(study.best_params)
        
        with open(f"{results_dir}/best_config.json", 'w') as f:
            json.dump(best_config, f, indent=2, default=str)
        
        # Save all trial results
        trials_data = []
        for trial in study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'user_attrs': trial.user_attrs,
                'state': trial.state.name
            }
            trials_data.append(trial_data)
        
        with open(f"{results_dir}/all_trials.json", 'w') as f:
            json.dump(trials_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to directory: {results_dir}")
        print("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
        print(f"📊 Completed trials: {len(study.trials)}")
        if study.trials:
            print(f"📈 Best value so far: {study.best_value:.4f}")
        return 1
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 

# for tests 
# python oh_old.py --dataset=ml-1m --n_trials=2 --storage=sqlite:///optuna.db --gpu=7 --model_type=custom --study_name=test_2


# python optimize_hyperparams.py --dataset=ml-1m --n_trials=3 --gpu=7 --model_type=custom --study_name=test_db --yes_wandb --storage
#####