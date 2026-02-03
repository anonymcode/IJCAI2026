
# ========= DATA-RELEATED =========
# data location
data_dir = 'preprocessed_data'
optuna_dir = 'optuna_logs'
optuna_db_path = 'optuna_db.db' # path to .db file
save_dir = 'saved_models'
wandb_dir = '' # path where wandb dir will be created
wandb_key= '' # wandb key

# data description
timeid = 'timestamp'
userid = 'userid'

# data preprocessing
use_cached_pcore = True # download ready-to-use amazon files instead of preprocessing
sequence_length_movies = 200
sequence_length_amazon = 50

# data splits
time_offset_q = 0.95
max_test_interactions = 50_000


# ========= MODEL-RELEATED =========
validation_interval = 20 # frequency of validation for iterative models; 1 means validate on each iteration


# ========= OPTUNA-RELEATED =========
grid_steps_limit = 60
disable_experimental_warnings = True
