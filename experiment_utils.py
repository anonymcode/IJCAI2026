##### imports #####
import time
import numpy as np
import scipy as sp
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.autograd import Function
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict
import gzip
import urllib
from ast import literal_eval
# from polara import get_movielens_data
import ssl
import requests
import time
import random
import defaults
from clearml import Task, Logger
from PIL import Image
import io
from scipy import stats

from math import sqrt

from utils import *
from models.SASRec_custom import SASRec, SharedKernel, Custom_A_FullMatrix
# from data import get_dataset


###################
# from utils import split_offsets
from processor import prepare_data, DataSet, prepare_data_exclude, prepare_data_cut




# Training and validating SASRec
# (we add option to create model with in Full CE mode)

def create_model(user_train, usernum, itemnum, config, maxlen, device, mode = 'BCE'):
    # Fix random seed for all sources using manual_seed from config
    manual_seed = config.get('manual_seed', 111) 
    
    model = SASRec(usernum, itemnum, config, maxlen, device).to(device)
    
    # Create deterministic generator for weight initialization
    init_generator = torch.Generator(device=device)
    init_generator.manual_seed(manual_seed)
    
    # Collect all parameters belonging to SharedKernel and Custom_A_FullMatrix
    special_params = set()
    skipped_params = []
    for module in model.modules():
        if isinstance(module, (SharedKernel, Custom_A_FullMatrix)):
            for param_name, param in module.named_parameters():
                special_params.add(param)
                skipped_params.append(f"{type(module).__name__}.{param_name}")

    print(f"[INIT] Skipping initialization for {len(special_params)} parameters: {skipped_params}")

    # Check SharedKernel parameter values BEFORE initialization
    for module in model.modules():
        if isinstance(module, SharedKernel):
            print(f"[INIT] SharedKernel parameters BEFORE general initialization:")
            print(f"  _g0: {module._g0.detach().cpu().numpy() if hasattr(module, '_g0') else 'N/A'}")
            print(f"  _g_rest[:5]: {module._g_rest.detach().cpu().numpy()[:5] if hasattr(module, '_g_rest') else 'N/A'}")
            print(f"  g[:5]: {module.g.detach().cpu().numpy()[:5]}")

    # Initialize ALL model parameters, EXCEPT those belonging to special modules
    initialized_count = 0
    for name, param in model.named_parameters():
        if param in special_params:
            continue  # Skip SharedKernel and Custom_A_FullMatrix parameters
        try:
            # TODO: try other initializations if needed
            torch.nn.init.xavier_uniform_(param.data, generator=init_generator)
            initialized_count += 1
        except Exception:
            # Ignore parameters to which Xavier is not applicable (e.g., bias=0, etc.)
            pass

    print(f"[INIT] Initialized {initialized_count} parameters with Xavier out of {len(list(model.named_parameters()))} total")

    # Check SharedKernel parameter values AFTER initialization
    for module in model.modules():
        if isinstance(module, SharedKernel):
            print(f"[INIT] SharedKernel parameters AFTER general initialization:")
            print(f"  _g0: {module._g0.detach().cpu().numpy() if hasattr(module, '_g0') else 'N/A'}")
            print(f"  _g_rest[:5]: {module._g_rest.detach().cpu().numpy()[:5] if hasattr(module, '_g_rest') else 'N/A'}")
            print(f"  g[:5]: {module.g.detach().cpu().numpy()[:5]}")
    
    # Use sampler_seed from config
    sampler_seed = config.get('sampler_seed', 99)
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=config['batch_size'], maxlen=maxlen, n_workers=0, seed=sampler_seed) # n_workers=0 for full determinism
    
    if mode == 'BCE':
        criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    elif mode == 'CE': 
        criterion = nn.CrossEntropyLoss()
    
#     if torch.cuda.is_available():
#         model = model.cuda()
#         criterion = criterion.cuda()

    #TODO try different betas
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98))
    return model, sampler, criterion, optimizer



def check_early_stop(target_score, previous_best, margin=0, max_attempts=10):
    margin = abs(margin)
    print(f"target_score: {target_score}")
    print(f"previous_best: {previous_best}")

    if target_score < previous_best + margin:
        print("we are counting because of not improving of the target metric")
        check_early_stop.fail_count += 1
        print(f"fail_count: {check_early_stop.fail_count}")
    else:
        check_early_stop.fail_count = 0
        print("we are resetting the counter because of improving of the target metric")
        print(f"fail_count: {check_early_stop.fail_count}")
    if check_early_stop.fail_count >= max_attempts:
        print('Interrupted due to early stopping condition.')
        print(f"fail_count: {check_early_stop.fail_count}")
        raise StopIteration



def evaluate_epoch(
        valid_dataset,
        epoch,
        loss,
        model,
        current_best_results,
        config,
        topn,
        silent = False,
        wandb_run = None,
        test_dataset = None,
        target_metric = None
        ):
    # validate model
    model.eval()
    validation_epoch_results = evaluate(model, valid_dataset, topn, config, eval_mode=config.get('val_eval_mode', 'successive'))
    
    # evaluate on test dataset if provided
    test_epoch_results = None
    # if test_dataset is not None:
    #     print(f"wtf")
    #     exit()
    #     model.save_attention_weights = True  # Enable attention saving
    #     test_epoch_results = evaluate(model, test_dataset, topn, config, eval_mode=config['test_eval_mode'])
    #     model.save_attention_weights = False # Disable back

    model.train()

    if target_metric is None: # target metric
        target_metric =f'NDCG@{topn}'

    
    previous_best_score = current_best_results[target_metric]['score'] # fix for early stopping
    if update_best(current_best_results, validation_epoch_results, target_metric):
        # config['epoch'] = epoch
        pass

    scores = {metric: res['score'] for metric, res in validation_epoch_results.items()}
    # if not silent:
    #     print("Validation scores:", scores)
        # if test_epoch_results is not None:
        #     test_scores = {metric: res_dict['score'] for metric, res_dict in test_epoch_results.items()}
        #     print("Test scores:", test_scores)

    # Logging images during evaluation
    
    # if wandb_run:
    #     try:
    #         # Take batch of sequences from test_dataset for attention averaging
    #         # This logic is no longer needed, as attention is saved in evaluate()


    #         log_dict = {}
    #         if fig1 is not None:
    #             log_dict["attention/A_W_d_T"] = get_wandb().Image(fig1)
    #         if fig2 is not None:
    #             log_dict["attention/A_V_emb_T"] = get_wandb().Image(fig2)
    #         if fig3 is not None:
    #             log_dict["attention/Layer0_Attn"] = get_wandb().Image(fig3)

    #         if log_dict:  # Add epoch label and log
    #             log_dict["epoch"] = epoch
    #             wandb_run.log(log_dict, step=epoch)

    #         # Close figures to avoid accumulation
    #         for fig in (fig1, fig2, fig3):
    #             if fig is not None:
    #                 plt.close(fig)
    #     except Exception as e:
    #         if not silent:
    #             print(f"Warning: Could not log attention matrices: {e}")

    # Log metrics to wandb

    val_log_dict = {f'val_{metric}': score for metric, score in scores.items()}
    # Log test metrics if available
    # if test_epoch_results is not None:
    #     test_scores = {metric: res_dict['score'] for metric, res_dict in test_epoch_results.items()}
    #     test_log_dict = {f'test_{metric}': score for metric, score in test_scores.items()} 
    #     val_log_dict.update(test_log_dict)

    if wandb_run:
        # Log validation metrics (each metric separately)
        for metric_name, metric_value in val_log_dict.items():
            wandb_run.report_scalar(
                title="Validation metrics",  # title of plot 
                series=metric_name,  # name of line in legend (each metric is a separate line)
                iteration=epoch,
                value=metric_value
            )

    # early stopping
    # check_early_stop(scores[target_metric], previous_best_score, margin=args.es_tol, max_attempts=args.es_max_steps)

    return scores, test_epoch_results

# we add option to run epoch in Full CE mode

def run_epoch(model, num_batch, l2_emb, sampler, optimizer, criterion, mode = 'BCE'):
    device = model.dev
    for _ in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
        # print(f"u.shape: {u.shape}")
        # print(f"seq.shape: {seq.shape}")
        # print(f"pos.shape: {pos.shape}")
        # print(f"neg.shape: {neg.shape}")

        # u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        
        if mode == 'BCE': 
            pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), torch.zeros(neg_logits.shape, device=device)
            optimizer.zero_grad()
            indices = np.where(pos != 0)            
            loss = criterion(pos_logits[indices], pos_labels[indices])
            loss += criterion(neg_logits[indices], neg_labels[indices])
            
        elif mode == 'CE': 
            optimizer.zero_grad()
            pos = torch.LongTensor(np.array(pos)).to(device)
            # print(f"pos.shape: {pos.shape}")
            indices = torch.where(pos.view(-1) != 0)
            preds = model(u, seq, pos, neg, mode = 'CE')
            batch_size = len(seq)
            n = preds.size(1)
            N = preds.size(-1)    
            # print(f"preds.shape: {preds.reshape(batch_size*n, N)[indices].shape}")
            # print(f"preds.shape: {pos.reshape(-1)[indices].shape}")

            loss = criterion(preds.reshape(batch_size*n, N)[indices],pos.reshape(-1)[indices])
            # # NEW VERSION
            # loss = criterion(
            #     preds.view(-1, preds.size(-1)),              # (B*n, N)
            #     pos.view(-1)                               # (B*n,)
            #     )        # elements equal to 0 are automatically ignored


            
        if l2_emb != 0:
            for param in model.item_emb.parameters():
                loss += l2_emb * torch.norm(param)
        loss.backward()
        optimizer.step()
    return loss


def sample_ci(scores, coef=2.776):
    """
    Computes confidence interval for a sample.
    
    Args:
        scores: list of values (e.g., metrics across trials)
        coef: t-distribution coefficient. If not specified, uses deprecated value 2.776.
              Recommended to use get_t_coefficient(len(scores)) for automatic computation.
    
    Returns:
        float: half-width of confidence interval (margin of error)
    
    Formula: CI = t_coef * (std / sqrt(n))
    """
    n = len(scores)
    if n < 2: # unable to estimate ci
        return np.nan
    return coef * np.std(scores, ddof=1) / sqrt(n)


METRICS = ['NDCG', 'MRR', 'HR', 'COV']

def hr_score(hit_idx):
    return 1.0

def mrr_score(hit_idx):
    return 1.0 / (hit_idx + 1)

def ndcg_score(hit_idx):
    # ideal DCG is 1/log2(2) = 1
    return 1.0 / np.log2(hit_idx + 2)




def evaluate_step(model, context_data, test_seq, topn, config=None):
    hr = []
    mrr = []
    ndcg = []
    unique_items = set()
    
    seen_test = defaultdict(list)
    for user, test_item in test_seq:
        if (user not in context_data) and (user not in seen_test):
            seen_test[user] = [test_item]
            continue # this is correct, we do not predict for user with ZERO history, we need at least one item in history

        seq = context_data.get(user, []) + seen_test.get(user, [])
        # seq = train.get(user, []) 
        hit_index, predicted_items = model.check_hit(test_item, seq, topn)
    
        try:
            hit_index = hit_index.item() # we expect only 1 item here
            # print(f"hit_index: {hit_index}")
#         except ValueError: # empty index or more then 1 item (which is incorrect) # raises not ValueError, but RuntimeError: a Tensor with 0 elements cannot be converted to Scalar
        except:
            hr_inc = mrr_inc = ndcg_inc = 0
        else:
            hr_inc = hr_score(hit_index)
            mrr_inc = mrr_score(hit_index)
            ndcg_inc = ndcg_score(hit_index)
        hr.append(hr_inc)
        mrr.append(mrr_inc)
        ndcg.append(ndcg_inc)

        seen_test[user].append(test_item) # extend seen items for next step prediction
        unique_items = unique_items.union(predicted_items)
    
    scores = {
        f'NDCG@{topn}': np.mean(ndcg),
        f'MRR@{topn}': np.mean(mrr),
        f'HR@{topn}': np.mean(hr),
        f'COV@{topn}': len(unique_items),
    }
    sqerrors = {
        f'NDCG@{topn}': np.mean((ndcg - scores[f'NDCG@{topn}'])**2) / (len(ndcg) - 1),
        f'MRR@{topn}': np.mean((mrr - scores[f'MRR@{topn}'])**2) / (len(mrr) - 1),
        f'HR@{topn}': np.mean((hr - scores[f'HR@{topn}'])**2) / (len(hr) - 1),
    }
    return scores, sqerrors

def evaluate_step_batched(model, context_data, test_seq, topn, batch_size=128, config=None):

    """Fast batched version of evaluate_step with true GPU vectorization.

    Main differences from previous implementation:
    1. For all users in batch, *one* sequence tensor of size (batch, maxlen) is formed
       and scores are computed in one model pass (instead of for loop over each user).
    2. Masking already seen items and top-k are performed on GPU.

    This drastically reduces the number of `model.score` calls and thereby speeds up
    inference.
    """

    device = model.dev

    # # Count: how many users from test_seq have history in context_data
    # unique_test_users = set()
    # for (user_id, test_item) in test_seq:
    #     unique_test_users.add(int(user_id))
    
    # users_with_history = 0
    # users_without_history = 0
    # for user_id in unique_test_users:
    #     if user_id in context_data and len(context_data[user_id]) > 0:
    #         users_with_history += 1
    #     else:
    #         users_without_history += 1
    
    # print(f"[DEBUG] Users from test_seq: total={len(unique_test_users)}, with_history={users_with_history}, without_history={users_without_history}")

    # Metric accumulators
    hr, mrr, ndcg = [], [], []
    unique_items = set()

    # For online update of user test item history
    seen_test = defaultdict(list)

    # Fix seeds (important for determinism with cuda)
    manual_seed = config.get('manual_seed', 111) if config is not None else 111
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)

    # Unpack (user, target) pairs from test sequence
    users, test_items = zip(*test_seq)
    users = np.asarray(users, dtype=np.int64)
    test_items = np.asarray(test_items, dtype=np.int64)

    maxlen = model.pos_emb.num_embeddings

    # Process in batches
    for start in range(0, len(users), batch_size):
        end = start + batch_size
        batch_users_np = users[start:end]
        batch_targets_np = test_items[start:end]

        batch_seqs = []     # actual sequences (list of list[int])
        valid_mask = []     # which batch elements are valid for inference

        # 1) form context sequences for each user
        for idx, (u, tgt) in enumerate(zip(batch_users_np, batch_targets_np)):
            u = int(u)
            tgt = int(tgt)
            if (u not in context_data) and (u not in seen_test):
                # user seen for first time and has no history - skip
                # print(f"[DEBUG] User {u}: no history, skipping target {tgt}")
                # print((u not in context_data))
                # print((u not in seen_test))
                seen_test[u] = [tgt]
                valid_mask.append(False)
                continue

            seq = context_data.get(u, []) + seen_test.get(u, []) # base history + seen history
            if len(seq) == 0:
                # print(f"[DEBUG] User {u}: empty history (base={context_data.get(u, [])}, seen={seen_test.get(u, [])}), skipping target {tgt}")
                # user has entry in context_data but history is empty - skip
                seen_test[u].append(tgt)
                valid_mask.append(False)
                continue

            # print(f"[DEBUG] User {u}: history_len={len(seq)}, history={seq[-10:] if len(seq) > 10 else seq}, target={tgt}")
            batch_seqs.append(seq)
            valid_mask.append(True)
            # update history for next step
            seen_test[u].append(tgt)

        if not any(valid_mask):
            print(f"ALARM: {batch_seqs}")
            # Entire batch is "empty" - move to next
            continue

        # 2) Prepare input tensor (B, maxlen)
        padded_seqs = np.zeros((len(batch_seqs), maxlen), dtype=np.int64)
        for j, seq in enumerate(batch_seqs):
            padded_seqs[j, -len(seq):] = seq[-maxlen:]

        # Convert to CPU int64 tensor; inside log2feats data will be
        # moved to the appropriate device.
        seq_tensor = torch.LongTensor(padded_seqs)

        # 3) Compute scores for all items for all users in batch at once
        with torch.no_grad():
            log_feats = model.log2feats(seq_tensor)      # (B, T, C)  already on device
            final_feat = log_feats[:, -1, :]             # (B, C)
            item_embs = model.item_emb.weight            # (I, C)
            predictions = torch.matmul(final_feat, item_embs.t())  # (B, I)

        # 4) Mask already seen items
        for j, seq in enumerate(batch_seqs):
            if len(seq) == 0:
                continue
            seen_tensor = torch.LongTensor(seq).to(device)
            predictions[j].index_fill_(0, seen_tensor, float('-inf'))

        # 5) Get top-k
        _, topk_indices = torch.topk(predictions, topn, dim=1)
        topk_np = topk_indices.cpu().numpy()

        # 6) Compute metrics
        valid_iter = (idx for idx, flag in enumerate(valid_mask) if flag)
        for j, orig_idx in enumerate(valid_iter):
            user_id = batch_users_np[orig_idx]
            target = batch_targets_np[orig_idx]

            pred_items = topk_np[j]
            unique_items.update(pred_items)

            hit_positions = np.where(pred_items == target)[0]
            if hit_positions.size > 0:
                hit_pos = int(hit_positions[0])
                hr.append(hr_score(hit_pos))
                mrr.append(mrr_score(hit_pos))
                ndcg.append(ndcg_score(hit_pos))
            else:
                hr.append(0.0)
                mrr.append(0.0)
                ndcg.append(0.0)

    # -- Final average metric values --
    hr_arr = np.asarray(hr)
    mrr_arr = np.asarray(mrr)
    ndcg_arr = np.asarray(ndcg)

    scores = {
        f'NDCG@{topn}': float(ndcg_arr.mean() if ndcg_arr.size else 0.0),
        f'MRR@{topn}': float(mrr_arr.mean() if mrr_arr.size else 0.0),
        f'HR@{topn}':  float(hr_arr.mean()  if hr_arr.size  else 0.0),
        f'COV@{topn}': len(unique_items),
    }

    # Standard errors (SEM) - square of SEM = s²/n, where s² = var(ddof=1)
    # We use var(ddof=0) / (n-1) = sum((x-mean)²) / (n*(n-1)) = correct square of SEM
    sqerrors = {
        f'NDCG@{topn}': float(ndcg_arr.var(ddof=0) / (len(ndcg_arr) - 1)) if len(ndcg_arr) > 1 else 0.0,
        f'MRR@{topn}': float(mrr_arr.var(ddof=0) / (len(mrr_arr) - 1)) if len(mrr_arr) > 1 else 0.0,
        f'HR@{topn}':  float(hr_arr.var(ddof=0)  / (len(hr_arr)  - 1)) if len(hr_arr)  > 1 else 0.0,
    }

    return scores, sqerrors

def evaluate(model, dataset, topn, config, eval_mode='last'):
    """
    Three evaluation modes are supported (config['eval_mode']):
      - 'last'       : predict only the last test item of each user,
                      all previous part of test_data is used as context.
      - 'random'     : for each user, a random target is selected from test_data,
                      context is train + test prefix up to selected target
                      (similar to evaluate_w_conf_intervals, but without bootstrap repeats).
      - 'successive' : sequential step-by-step prediction of all test items
                      of user in chronological order.
    If test_data is passed as pd.Series (old format), the previous
    "successive" step-by-step logic is used.
    """
    context_raw, test_raw, usernum, itemnum = dataset
    # eval_mode = (config or {}).get('eval_mode', 'successive')

    # def _normalize_seq_map(data):
    #     normalized = {}
    #     if data is None:
    #         return normalized
    #     for u, seq in data.items():
    #         if seq is None:
    #             normalized[u] = []
    #         elif isinstance(seq, list):
    #             normalized[u] = seq
    #         else:
    #             normalized[u] = list(seq)
    #     return normalized

    def _accumulate(step_scores, step_sqerr, results_store):
        for metric, score in step_scores.items():
            if metric.startswith('COV'):
                score = float(score) / itemnum
                error = None  # COV error is calculated separately
            else:
                error = step_sqerr.get(metric, 0.0)
            results_store[metric]['scores'].append(score)
            results_store[metric]['squared_errors'].append(error)

    results = defaultdict(lambda: defaultdict(list))  # {metric: {'scores': [], 'squared_errors': []}}

    # ---- Modern format: dict user_id -> list[item_id] ----
    if isinstance(test_raw, dict):
        context_data = context_raw
        test_data = test_raw
        mode = eval_mode
        batch_size = int((config or {}).get('eval_batch_size', 256))

        eval_context = {}
        eval_pairs = []

        if mode == 'last': # correct
            for u, items in test_data.items():
                if not items:
                    continue

                prefix, target = items[:-1], items[-1]
                base_hist = context_data.get(u, [])
                if len(base_hist) + len(prefix) == 0:
                    continue  # nothing to predict without history
                eval_context[u] = base_hist + prefix
                eval_pairs.append((u, target))

        elif mode == 'random':
            base_seed = (config or {}).get('eval_random_seed', (config or {}).get('manual_seed', 111))
            num_trials = int((config or {}).get('eval_random_trials', 5))
            any_pairs = False

            # List of results by trials:
            # metric -> [score_trial_0, score_trial_1, ..., score_trial_{K-1}]
            trials_scores = defaultdict(list)

            for trial in range(num_trials):
                rng = np.random.RandomState(base_seed + trial)
                eval_context = {}
                eval_pairs = []

                for u, items in test_data.items():
                    if not items:
                        continue
                    idx = 0 if len(items) == 1 else rng.randint(0, len(items))

                    prefix, target = items[:idx], items[idx]
                    base_hist = context_data.get(u, [])
                    if len(base_hist) + len(prefix) == 0:
                        continue  # no history to predict
                    eval_context[u] = base_hist + prefix
                    eval_pairs.append((u, target))

                if len(eval_pairs) == 0:
                    continue

                any_pairs = True
                step_scores, step_sqerr = evaluate_step_batched(
                    model,
                    eval_context,
                    eval_pairs,
                    topn,
                    batch_size=batch_size,
                    config=config,
                )

                # Save metric values for current trial
                for metric, score in step_scores.items():
                    # For COV normalize same as in _accumulate
                    if metric.startswith('COV'):
                        score = float(score) / itemnum
                    trials_scores[metric].append(float(score))

            if not any_pairs:
                print("[EVAL] No valid examples for evaluation (random).")
                return defaultdict(dict)

            # Average over num_trials and compute error as standard error of mean
            averaged_results = defaultdict(dict)
            for metric, scores_list in trials_scores.items():
                if not scores_list:
                    averaged_results[metric]['score'] = 0.0
                    averaged_results[metric]['error'] = 0.0
                    continue

                scores_array = np.array(scores_list, dtype=float)
                mean_val = float(scores_array.mean())

                if scores_array.size > 1:
                    std_val = float(scores_array.std(ddof=1))
                    stderr_val = std_val / np.sqrt(scores_array.size)
                else:
                    stderr_val = 0.0

                averaged_results[metric]['score'] = mean_val
                averaged_results[metric]['error'] = stderr_val # still SEM between trials, not between steps

            return averaged_results

        elif mode == 'successive':
            eval_context = context_data
            # User order doesn't matter - main thing is that for each user
            # their test items go in chronological order (guaranteed by
            # test_data[u] structure as a list)
            for u, items in test_data.items(): # for loop over users
                if not items:
                    continue

                for it in items: # for loop over items
                    eval_pairs.append((u, it))
        else:
            raise ValueError(f"Invalid evaluation mode: {mode}")

        if mode != 'random':  # random already accumulated num_trials results
            if len(eval_pairs) == 0:
                print("[EVAL] No valid examples for evaluation.")
                return defaultdict(dict)

            step_scores, step_sqerr = evaluate_step_batched(
                model,
                eval_context,
                eval_pairs,
                topn,
                batch_size=batch_size,
                config=config,
            )
            print(f"1step_scores: {step_scores}")
            print(f"1step_sqerr: {step_sqerr}")
            # step_scores2, step_sqerr2 = evaluate_step(
            #     model,
            #     eval_context,
            #     eval_pairs,
            #     topn,
            #     config=config,
            # )
            # print(f"2step_scores2: {step_scores2}")
            # print(f"2step_sqerr2: {step_sqerr2}")
            _accumulate(step_scores, step_sqerr, results)
            print("total number of items: ", itemnum)
            print("bla",step_scores['COV@10']/itemnum)

    # ---- Old format: Series with step -> [(user, item), ...] ----
    # else:
    #     test_data = test_raw
    #     if isinstance(test_data, (list, tuple)):
    #         test_data = pd.Series({0: test_data})
    #     context_data = _normalize_seq_map(context_raw)
    #     for step, test_seq in test_data.items():
    #         step_scores, step_sqerr = evaluate_step_batched(
    #             model,
    #             context_data,
    #             test_seq,
    #             topn,
    #             batch_size=256,
    #             config=config,
    #         )
    #         _accumulate(step_scores, step_sqerr, results)

    averaged_results = defaultdict(dict)  # {metric: {'score': _, 'error': _}}
    for metric, res in results.items():
        scores_list = res['scores']
        averaged_results[metric]['score'] = float(np.mean(scores_list)) if scores_list else 0.0
        if metric.startswith('COV'):
            averaged_results[metric]['error'] = sample_ci(scores_list)
        else:
            sqerrs = [e for e in res['squared_errors'] if e is not None]
            averaged_results[metric]['error'] = (
                float(sqrt(sum(sqerrs)) / len(sqerrs)) if len(sqerrs) > 0 else 0.0
            )

    return averaged_results



def update_best(best_results, results, target_metric):
#     print(best_results[target_metric]['score'])
#     print(results[target_metric])
    updated = results[target_metric]['score'] > best_results[target_metric]['score']
    if updated:
        best_results.update(results)
    return updated









def run_experiment(model,
                   dataset,
                   sampler, 
                   criterion, 
                   optimizer, 
                   param_config, 
                   topn = 10, 
                   num_epochs = 200,
                   device = 'cuda',
                   mode= "CE",
                   silent = False,
                   early_stop_decline_streak = 10,
                   evaluate_on_every= 10,
                   valid_dataset = None,
                   test_dataset = None,
                   target_metric =f'NDCG@10',
                   wandb_run = None,
                   ):
    
    # early_stop_decline_streak - the number of evalueated epochs with declining target metric (NDCG@10), crtiterion to do early stopping
    early_stop = False
    if early_stop_decline_streak is not None:
        early_stop = True

    if target_metric is None:
        target_metric =f'NDCG@{topn}'
    test_metric_results = None


#     dataset, _ = read_dataset('ml-1m', "4months-4months", stepwise_eval=False, part='validation')
#     dataset, _ = read_dataset('amz-b', "3weeks-3weeks", stepwise_eval=False, part='validation')
    
    # display_stats(dataset)
    user_train, _, usernum, itemnum = dataset

    maxlen  = param_config['maxlen']
    num_batch = len(user_train) // param_config['batch_size'] # tail? + ((len(user_train) % args.batch_size) != 0)
    l2_emb = param_config['l2_emb']
    model.train() # enable model training

    model_best_results = defaultdict(lambda: defaultdict(lambda: -np.inf)) # {metric: {'score': _, 'error': _}}
    best_model_state = None  # for saving the best model state
    epoch_start_idx = 1

    losses = []

    metric_results = dict()
    for metric_name in METRICS:
        metric_results[f'{metric_name}@{topn}'] = []

    eval_epochs = []
    check_early_stop.fail_count = 0

    for epoch in  tqdm(range(epoch_start_idx, num_epochs + 1)): # main loop
        # time_start = time.time()
        loss = run_epoch(model, num_batch, l2_emb, sampler, optimizer, criterion, mode = mode)
        losses.append(loss.item())
        # print(f"time taken to run_epoch: {time.time() - time_start}") # 11
        
        # Log training loss every epoch
        if wandb_run:
            wandb_run.report_scalar(
                title="Train loss", # title of plot 
                series="train_loss", # name of line in legend
                iteration=epoch,
                value=loss.item()
            )
        
        if epoch % evaluate_on_every == 0:
            try:
                previous_best_score = model_best_results[target_metric]['score'] # fix for early stopping

                scores, test_epoch_results = evaluate_epoch(
                    valid_dataset,
                    epoch,
                    loss,
                    model,
                    model_best_results,
                    param_config,
                    topn,
                    silent,
                    wandb_run,
                    test_dataset=None,
                    target_metric=target_metric
                    )
                # print(f"Time taken for evaluation: {time.time() - time_start_eval:.2f} seconds")
                for metric_name in scores.keys():
                    metric_results[metric_name].append(scores[metric_name])
                if scores[target_metric] > previous_best_score: # if new best score, save the model state
                    print("updated best_model_state")
                    best_model_state = model.state_dict().copy()
                
                # Collect test metrics if available
                if test_epoch_results is not None:
                    # Initialize test metrics if test_dataset is provided
                    test_metric_results = dict()
                    for metric_name in METRICS:
                        test_metric_results[f'{metric_name}@{topn}'] = []

                    test_scores = {metric: res_dict['score'] for metric, res_dict in test_epoch_results.items()}
                    for metric_name in test_scores.keys():
                        test_metric_results[metric_name].append(test_scores[metric_name])
                    
                eval_epochs.append(epoch)
                
                if early_stop:
                    check_early_stop(scores[target_metric], previous_best_score, margin=0, max_attempts=early_stop_decline_streak)

            except StopIteration: # early stopping condition met
                print('EARLY STOPPING')
                break

    # load the best model state
    if best_model_state is not None: 
        model.load_state_dict(best_model_state)
    else:
        print("ALARM: best_model_state is None")
        exit()

    # Log final best model metrics to wandb
    if wandb_run:
        if eval_epochs and metric_results and target_metric in metric_results:
            # Find the epoch with the best target_metric on validation
            best_epoch_idx = np.argmax(metric_results[target_metric])
            
            best_model_metrics = {}
            
            # Log validation metrics from the best epoch
            for metric_name, values in metric_results.items():
                if values and best_epoch_idx < len(values):
                    best_model_metrics[f'best_model/val_{metric_name}'] = values[best_epoch_idx]
            
            # Log test metrics from the same epoch as best validation
            # if test_metric_results:
            #     for metric_name, values in test_metric_results.items():
            #         if values and best_epoch_idx < len(values):
            #             best_model_metrics[f'best_model/test_{metric_name}'] = values[best_epoch_idx]
            if test_dataset is not None: # evaluate on test dataset
                # print("we are here")
                # Initialize test metrics if test_dataset is provided
                test_metric_results = dict()
                for metric_name in METRICS:
                    test_metric_results[f'{metric_name}@{topn}'] = [] # empty list

                # --- final metric computation on test set (best model state) ---
                model.eval()
                model.save_attention_weights = True  # Enable attention saving
                final_test_results = evaluate(model, test_dataset, topn, param_config, eval_mode=param_config.get('test_eval_mode', 'successive'))
                model.save_attention_weights = False  # Disable attention saving
                print(f"[CI-EVAL]⚠️ final_test_results: {final_test_results}") # ⚠️ final_test_results: defaultdict(<class 'dict'>, {'NDCG@10': {'score': 0.006410517327420551, 'error': 0.0003565151203016462}, 'MRR@10': {'score': 0.004490550221002895, 'error': 0.00030359287976056354}, 'HR@10': {'score': 0.012808641975308644, 'error': 0.0006344307270233201}, 'COV@10': {'score': 0.0073079915527398025, 'error': 2.7787040124485803e-05}})

                create_attention_figures(model, border=(param_config['maxlen'] - 40, param_config['maxlen']), wandb_run=wandb_run)

                # save results for return from function
                for metric, res in final_test_results.items():
                    score = res['score']
                    wandb_run.report_single_value(name=f'test_{metric}', value=score)
                    best_model_metrics[f'best_model/test_{metric}'] = score
                    error = res.get('error', None)
                    if error is not None and not (isinstance(error, float) and np.isnan(error)):
                        error = format(float(error), '.12f')
                        wandb_run.report_single_value(name=f'test_{metric}_error', value=error)
                        best_model_metrics[f'best_model/test_{metric}_error'] = error
                    # pack as list to maintain compatibility with existing structure
                    test_metric_results[metric] = [score]

    return losses, eval_epochs, metric_results, model, test_metric_results,

def plot_metric_results(eval_epochs, dict_metric_results): 
    # comparing results of different models across evaluation of different epochs
#     eval_epochs = np.arange(1, 20)*10
    
    num_epochs = len(eval_epochs)
    
    model_names = list(dict_metric_results.keys())
    num_of_models = len(model_names)
    metric_names = list(dict_metric_results[model_names[0]].keys())
    num_of_metrics = len(metric_names)
    
    figure, axis = plt.subplots(num_of_metrics, 1, figsize=(8, 10)) 
    figure.tight_layout(pad=5.0)
    for i in range(num_of_metrics):
        for j in range(num_of_models):
            model_name = model_names[j]
            metric_name = metric_names[i]
            
#             print(model_name)
#             print(metric_name)
#             print(dict_metric_results[model_name][metric_name])

            curr_metrics = dict_metric_results[model_name][metric_name]
            if len(curr_metrics) < num_epochs:
                diff = [None]*(num_epochs - len(curr_metrics))
                curr_metrics.extend(diff)
            else:
                curr_metrics = curr_metrics[:num_epochs]
            
            axis[i].plot(eval_epochs, curr_metrics, label=model_name) 
            axis[i].set_title(metric_name)
            axis[i].legend()
            
    plt.show() 



################# TEST INFERENCE #######################

def test_model_factory(dataset, config, device):
    display_stats(dataset)
    user_train, _, usernum, itemnum = dataset
    assert isinstance(user_train, dict)
    
    maxlen = config['maxlen']
    model, sampler, criterion, optimizer = create_model(user_train, usernum, itemnum, config, maxlen, device, mode = config['mode'])

    # model, sampler, criterion, optimizer = create_model(user_train, usernum, itemnum, config, maxlen, device, seed=88, mode= "CE")

    
    num_batch = len(user_train) // config['batch_size'] # tail? + ((len(user_train) % args.batch_size) != 0)
    l2_emb = config['l2_emb']
    model.train() # enable model training
    num_epochs = config['num_epochs']
    for _ in tqdm(range(num_epochs)):
        run_epoch(model, num_batch, l2_emb, sampler, optimizer, criterion, mode=config['mode'])
    sampler.close()
    
    model.eval()
    return model




######################### MY CODE #########################


def fix_random_seeds(manual_seed):
    """
    Globally fixes all sources of randomness.
    
    Args:
        manual_seed: seed for PyTorch, CUDA and NumPy
        sampler_seed: seed for WarpSampler (if None, uses manual_seed)
    """
    # PyTorch
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
    
    # NumPy
    np.random.seed(manual_seed)
    
    # Python random
    # random.seed(manual_seed)
    
    return

def fix_random_seeds(manual_seed):
    """
    Globally fixes all sources of randomness.
    
    Args:
        manual_seed: seed for PyTorch, CUDA and NumPy
        sampler_seed: seed for WarpSampler (if None, uses manual_seed)
    """
    # PyTorch
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)
    
    # NumPy
    np.random.seed(manual_seed)
    
    # Python random
    # random.seed(manual_seed)
    
    return



def seed_everything(seed=42):
    """
    Maximum strictness in fixing all sources of randomness for complete determinism
    """
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Stricter settings for complete determinism
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # Important: these settings may slow down training, but guarantee determinism
        
    # NumPy
    np.random.seed(seed)
    
    # Python random (if used)
    # import random
    # random.seed(seed)
    
    # Set environment variable for hashing
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For PyTorch >= 1.12 can also set
    # torch.use_deterministic_algorithms(True)
    # But this may cause errors with some operations



def train_and_eval(dataset_name,
                    topn=10,
                    param_config=None,
                    device='cuda',
                    verbose=False,
                    experiment_name=None,
                    use_wandb=True,
                    ):
    """
    """
    # Fix all seeds globally
    # fix_random_seeds(
    #     manual_seed=param_config.get('manual_seed', 42),
    # )
    seed_everything(param_config.get('manual_seed', 42))
    
    maxlen = param_config['maxlen']
    target_metric =f'NDCG@{topn}'

# My splits: 
    #for ml-1m [0.95, 0.97] is good split # train_data, validation_data, test_data: 920208, 49966, 29988
    #for ml-10m [0.95, 0.97] is good split # train_data, validation_data, test_data: 9920053, 49364, 29716
    #for yelp [0.95, 0.97] is good split # train_data, validation_data, test_data: 2214803, 47378, 28727
    #for gowalla [0.95, 0.97] is good split # train_data, validation_data, test_data: 2547662, 49426, 29862

    # tune_datapack, test_datapack = prepare_data(dataset_name, time_offset_q=[0.95, 0.97]) # split 0.95 example or 50k interactions
    tune_datapack, test_datapack = prepare_data_exclude(dataset_name, time_offset_q=[0.95, 0.97]) # split 0.95 example or 50k interactions
    # tune_datapack, test_datapack = prepare_data_cut(dataset_name, time_offset_q=[0.95, 0.97]) # 

    userid, itemid, timeid = 'userid', 'itemid', 'timestamp' # names of columns in the data

    train_df,  valid_df, tune_datapack_index = tune_datapack      # train + valid
    train_full_df, test_df, test_datapack_index = test_datapack    # (train+valid) + test
    # print(f"tune_datapack_index: {tune_datapack_index.keys()}") # ['users', 'items', 'new_users']
    # print(f"test_datapack_index: {test_datapack_index.keys()}") # ['users', 'items', 'new_users']

    # print(f"tune_datapack_index: {tune_datapack_index['items']}") # length=135369
    # print(f"test_datapack_index: {test_datapack_index['items']}") # length=136547


    for df in (train_df, valid_df, train_full_df, test_df):
        df[itemid] = df[itemid] + 1

    # tune_datapack, test_datapack
    # print(f"tune_datapack: {type(tune_datapack)}") # tuple
    # print(f"test_datapack: {type(test_datapack)}") # tuple
    # print(f"tune_datapack: {len(tune_datapack)}") # 3: train_data, test_data, data_index
    # print(f"test_datapack: {len(test_datapack)}") # 3: train_data, test_data, data_index

    # print(f"train_data: {len(tune_datapack[0])}") 
    # print(f"train_data + validation_data: {len(test_datapack[0])}")
    # print(f"validation_data: {len(tune_datapack[1])}")
    # print(f"test_data: {len(test_datapack[1])}")



    # ---- convert to format [user_train, holdout, n_users, n_items] ----
    dataset_valid = generate_partition(train_df,      # history before validation
                                    valid_df,      # one hold-out
                                    userid, itemid, timeid,
                                    stepwise_eval=False)

    dataset_test  = generate_partition(train_full_df, # history before test
                                    test_df,       # hold-out for test
                                    userid, itemid, timeid,
                                    stepwise_eval=False)

    # for training/validation use the same set
    train_dataset  = dataset_valid
    valid_dataset  = dataset_valid
    test_dataset   = dataset_test

    # unpack needed parts
    user_train_only, user_valid_list, usernum_global, itemnum_global = train_dataset
    user_train_full, user_test_list, usernum_testpack, itemnum_testpack = test_dataset

    # print(f"usernum_testpack: {itemnum_global}") # 135369
    # print(f"itemnum_testpack: {itemnum_testpack}") # 136547
    # exit(0)
    wandb_run = None
    if use_wandb:
        # wandb_config = param_config.copy()
        # wandb_config.update({
        #     'model_type': 'SASRec',
        #     'dataset_stats': {
        #         'usernum': usernum,
        #         'itemnum': itemnum,
        #         'num_batches': num_batch
        #     }
        # })
        # Create experiment name
        if experiment_name:
            run_name = experiment_name
        else:
            # run_name = f"kek-SASRec-{mode}-{'CustomA' if param_config.get('use_custom_A', False) else 'noCustomA'}-{'noPosEmb' if not param_config.get('use_pos_emb', True) else 'PosEmb'}"
            run_name = "name_hz"
        # wb_key = defaults.wandb_key
        # print(wb_key)
        os.environ["CLEARML_WEB_HOST"] = "http://178.170.194.157:8080/"
        os.environ["CLEARML_API_HOST"] = "http://178.170.194.157:8008"
        os.environ["CLEARML_FILES_HOST"] = "http://178.170.194.157:8081"
        os.environ["CLEARML_API_ACCESS_KEY"] = "WSDEDMR40A0QO7R5AO0HPZ7MGD0NFQ"
        os.environ["CLEARML_API_SECRET_KEY"] = "DfHnfhvHaHWPwBJ2Ty4PaDEscgPi-jODJ8T2YjxSn7QwBaK9CCpRHtwL9WiitveIK4U"

        # # 2. Project setup
        # os.environ["WANDB_PROJECT"] = "SASRec-train-eval" 
        
        # print(os.environ["WANDB_API_KEY"])
        # print(os.environ["WANDB_PROJECT"])

        # wandb_run = get_wandb().init(
        #     project=project_name,
        #     config=wandb_config,
        #     name=run_name,
        #     dir=defaults.wandb_dir, # path where wandb dir will be created
        # )
        task =Task.init(
            project_name="SASRec-train-eval",
            task_name=run_name,
            # config=wandb_config
            )

        # Log entire param_config in ClearML as hyperparameters before training start
        try:
            if isinstance(param_config, dict):
                # each key-value pair from dict becomes a separate task parameter
                task.connect(param_config, name="param_config")
            else:
                # in case something other than dict comes here someday
                task.connect({"param_config": param_config})
        except Exception as e:
            print(f"Warning: failed to log param_config to ClearML: {e}")

        # wandb_run = Task.get_logger()
        wandb_run = Logger.current_logger()


    model, sampler, criterion, optimizer = create_model(
        user_train_only,
        usernum_global,
        itemnum_global + 1,
        param_config,
        maxlen,
        device,
        mode=param_config['mode']
    )


    losses, eval_epochs, metric_results, model, test_metric_results = run_experiment(
        model=model,
        dataset=train_dataset,
        sampler=sampler,
        criterion=criterion,
        optimizer=optimizer,
        param_config=param_config,
        topn=topn,
        num_epochs=param_config['num_epochs'],
        device=device,
        mode=param_config['mode'],
        silent=not verbose,
        early_stop_decline_streak=param_config['early_stop_decline_streak'],
        evaluate_on_every=param_config['evaluate_on_every'],
        # use_wandb=use_wandb,
        # project_name="SASRec-train-eval",
        # experiment_name=experiment_name,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
        target_metric=target_metric,
        wandb_run=wandb_run,
        # task=task,
    )
    # save model if needed
    if param_config.get('save_model', False):
        os.makedirs(defaults.save_dir, exist_ok=True)
        study_name = param_config.get('study_name', 'ERROR_unknown_study_name')
        model_filename = f"{defaults.save_dir}/{study_name}_{experiment_name}.pth"
        torch.save(model.state_dict(), model_filename)
        print(f"Model saved to {model_filename}")

    # bucket evaluation
    if param_config.get('test_eval_buckets', False):
        # Evaluate performance across popularity buckets on the test set
        eval_bucket_scores = eval_buckets(
            model=model,
            test_dataset=test_dataset,
            train_df=train_full_df,
            itemid_col=itemid,
            topn=topn,
            config=param_config,
            wandb_run=wandb_run,
            verbose=verbose
        )

        # --- Log full bucket score dictionary to WandB under a dedicated panel ---
        if wandb_run is not None:
            bucket_names = list(eval_bucket_scores.keys())
            
            # Get list of all metrics from first bucket
            if bucket_names:
                metric_names = list(eval_bucket_scores[bucket_names[0]].keys())
                
                # For each metric create separate histogram
                for metric_name in metric_names:
                    # Extract values for this metric from all buckets
                    values = []
                    bucket_labels = []
                    
                    for bucket_name in bucket_names:
                        if metric_name in eval_bucket_scores[bucket_name]:
                            # Convert numpy.float64 to regular float
                            value = float(eval_bucket_scores[bucket_name][metric_name])
                            values.append(value)
                            bucket_labels.append(bucket_name)

                    print(f"metric_name: {metric_name}")
                    print(f"values: {values}")
                    print(f"labels: {bucket_labels}")

                    if values:  # Check that there is data for logging
                        # Convert to 2D array: each row is one data series
                        values_2d = np.array(values).reshape(1, -1)
                        print(f"values_2d: {values_2d}")
                        wandb_run.report_histogram(
                            title=f"Bucket Performance - {metric_name}",
                            series=metric_name,
                            values=values_2d,
                            iteration=0,
                            labels=[metric_name],  # One label for one row of data
                            xlabels=bucket_labels,  # Labels for each bar on x-axis
                            xaxis="Popularity Buckets",
                            yaxis=metric_name
                        )
    sampler.close()
    
    # # === Sample recommendations for a few users ===
    # sample_size = 4  # how many users to show
    # model.eval()

    # def _recommend_for_users(user_ids, user_hist_dict, split_name):
    #     if len(user_ids) == 0:
    #         return
    #     # Choose up to sample_size unique user ids
    #     chosen_uids = random.sample(user_ids, k=min(sample_size, len(user_ids)))
    #     print(f"\n{split_name} split – top{topn} recommendations for {len(chosen_uids)} users:")
    #     for uid in chosen_uids:
    #         history = user_hist_dict.get(uid, [])
    #         if len(history) == 0:
    #             print(f"User {uid}: no history, skipped")
    #             continue
    #         seq_tensor = torch.LongTensor(history).to(model.dev)
    #         with torch.no_grad():
    #             preds = model.score(seq_tensor)
    #         # Mask already seen items
    #         if seq_tensor.numel() > 0:
    #             preds = preds.clone()
    #             preds.index_fill_(0, seq_tensor, float('-inf'))
    #         _, top_items = torch.topk(preds, topn)
    #         print(f"User {uid}: {top_items.cpu().numpy().tolist()}")

    # # Collect distinct user ids from validation and test target lists
    # val_user_ids = list({u for u, _ in user_valid_list})
    # test_user_ids = list({u for u, _ in user_test_list})

    # _recommend_for_users(val_user_ids, user_train_only, "Validation")
    # _recommend_for_users(test_user_ids, user_train_full, "Test")

    print('training with validation completed')
    
    # Pack training results
    training_results = {
        'losses': losses,
        'eval_epochs': eval_epochs,
        'metric_results': metric_results,
        'best_epoch': eval_epochs[np.argmax(metric_results[f'HR@{topn}'])] if metric_results[f'HR@{topn}'] else None,
        f'best_{target_metric}': max(metric_results[target_metric]) if metric_results[target_metric] else None
    }
    
    # For Optuna extract best validation metrics from already computed results
    validation_scores = {}
    validation_errors = {}
    
    if metric_results and eval_epochs:
        # Find epoch with best result by target metric
        if target_metric in metric_results and metric_results[target_metric]:
            best_epoch_idx = np.argmax(metric_results[target_metric])
            
            # Extract all validation metrics from best epoch
            for metric_name, values in metric_results.items():
                if values and best_epoch_idx < len(values):
                    validation_scores[metric_name] = values[best_epoch_idx]
                    validation_errors[metric_name] = 0.0  # Errors not used for optimization
        else:
            # If no target metric, take last values
            for metric_name, values in metric_results.items():
                if values:
                    validation_scores[metric_name] = values[-1]
                    validation_errors[metric_name] = 0.0

    # if hasattr(get_wandb(), 'run') and get_wandb().run is not None:
    #     get_wandb().finish()
    if wandb_run:
        task.close()
    
    return training_results, validation_scores, validation_errors, model
    # return

def banded(g):
    n=len(g) # 200
    # print("len of g", n) 
    T = np.zeros((n, n)) # 200x200
    for x in range(n):
        T[x][x:x+n]=g[:n-x]
    return T.T

def create_attention_figures(model, border: tuple = (0,50), wandb_run=None):
    
    if wandb_run is None:
        return
    """
    Logs to wandb/clearML Custom_A matrices (if present) and saved attention
    weights. Used only for visualization after final evaluation.
    """
    # --- 1. Custom_A Matrices ---------------------------------------------------
    try:
        # New variant: multiple A1/A2 layers in each block-cell
        if hasattr(model, 'custom_A1_layers') and hasattr(model, 'custom_A2_layers') \
           and len(model.custom_A1_layers) == len(model.custom_A2_layers) \
           and len(model.custom_A1_layers) > 0:

            A1_mats = []  # list of restored A1 matrices by layers
            A2_mats = []  # list of restored A2 matrices by layers

            for layer_idx, (A1_layer, A2_layer) in enumerate(zip(model.custom_A1_layers, model.custom_A2_layers)):
                # Restore full A1/A2 matrices depending on block type.
                # 1) Conv variant (Custom_A_Conv): extract convolution kernel and build Toeplitz matrix.
                # 2) FullMatrix variant (Custom_A_FullMatrix): take parameter A with triangular mask.

                A1_full = None
                A2_full = None

                # --- A1 ---
                if hasattr(A1_layer, 'conv'):
                    # Conv variant (old behavior)
                    w1 = torch.squeeze(A1_layer.conv.weight).detach().cpu().numpy()
                    is_upper_A1 = getattr(A1_layer, 'upper_triangular', False)

                    # Log convolution vector in clearML
                    wandb_run.report_histogram(
                        title=f"A1 Conv Vector Layer {layer_idx}",
                        series=f"A1_conv_vector_L{layer_idx}",
                        values=w1.reshape(1, -1),
                        iteration=getattr(model, 'batch_counter', 0),
                        labels=["conv_weights"],
                        xaxis="Position",
                        yaxis="Weight Value"
                    )
                    print(f"w1 {layer_idx}: {w1}")

                    if not is_upper_A1:
                        w1 = np.flip(w1, axis=0)
                    A1_full = banded(w1)
                    if is_upper_A1:
                        A1_full = A1_full.T
                elif hasattr(A1_layer, 'A'):
                    # FullMatrix variant
                    A1_param = A1_layer.A.detach().cpu().numpy()
                    mask1 = getattr(A1_layer, 'triangular_mask', None)
                    if mask1 is not None:
                        mask1 = mask1.detach().cpu().numpy()
                        A1_full = A1_param * mask1
                    else:
                        A1_full = A1_param

                # --- A2 ---
                if hasattr(A2_layer, 'conv'):
                    w2 = torch.squeeze(A2_layer.conv.weight).detach().cpu().numpy()
                    is_upper_A2 = getattr(A2_layer, 'upper_triangular', False)

                    # Log convolution vector in clearML
                    wandb_run.report_histogram(
                        title=f"A2 Conv Vector Layer {layer_idx}",
                        series=f"A2_conv_vector_L{layer_idx}",
                        values=w2.reshape(1, -1),
                        iteration=getattr(model, 'batch_counter', 0),
                        labels=["conv_weights"],
                        xaxis="Position",
                        yaxis="Weight Value"
                    )
                    print(f"w2 {layer_idx}: {w2}")

                    if not is_upper_A2:
                        w2 = np.flip(w2, axis=0)
                    A2_full = banded(w2)
                    if is_upper_A2:
                        A2_full = A2_full.T
                elif hasattr(A2_layer, 'A'):
                    A2_param = A2_layer.A.detach().cpu().numpy()
                    mask2 = getattr(A2_layer, 'triangular_mask', None)
                    if mask2 is not None:
                        mask2 = mask2.detach().cpu().numpy()
                        A2_full = A2_param * mask2
                    else:
                        A2_full = A2_param

                # If for some reason failed to restore matrix – skip layer
                if A1_full is None or A2_full is None:
                    continue

                A1_mats.append(A1_full)
                A2_mats.append(A2_full)

                # Visualize and log each layer
                fig1, ax1 = plt.subplots(figsize=(8, 6))
                im1 = ax1.imshow(A1_full[border[0]:border[1], border[0]:border[1]], interpolation='none')
                plt.colorbar(im1, ax=ax1)
                orient1 = "upper" if getattr(A1_layer, 'upper_triangular', False) else "lower"
                ax1.set_title(f"A1[L{layer_idx}] [{orient1}] (min={A1_full.min():.6f}, max={A1_full.max():.6f})")
                buf1 = io.BytesIO()
                fig1.savefig(buf1, format='jpg', bbox_inches='tight', dpi=100)
                buf1.seek(0)
                pil_img1 = Image.open(buf1)
                plt.close(fig1)

                wandb_run.report_image(
                    title=f"A1 Layer {layer_idx}",
                    series="A1_per_layer",
                    iteration=getattr(model, 'batch_counter', 0),
                    image=pil_img1
                )

                fig2, ax2 = plt.subplots(figsize=(8, 6))
                im2 = ax2.imshow(A2_full[border[0]:border[1], border[0]:border[1]], interpolation='none')
                plt.colorbar(im2, ax=ax2)
                orient2 = "upper" if getattr(A2_layer, 'upper_triangular', False) else "lower"
                ax2.set_title(f"A2[L{layer_idx}] [{orient2}] (min={A2_full.min():.6f}, max={A2_full.max():.6f})")
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format='jpg', bbox_inches='tight', dpi=100)
                buf2.seek(0)
                pil_img2 = Image.open(buf2)
                plt.close(fig2)

                wandb_run.report_image(
                    title=f"A2 Layer {layer_idx}",
                    series="A2_per_layer",
                    iteration=getattr(model, 'batch_counter', 0),
                    image=pil_img2
                )

                # Compute and log C = A1^T * A2 for each layer
                C_full = A1_full.T @ A2_full
                figC, axC = plt.subplots(figsize=(8, 6))
                imC = axC.imshow(C_full[border[0]:border[1], border[0]:border[1]], interpolation='none')
                plt.colorbar(imC, ax=axC)
                axC.set_title(f"C[L{layer_idx}] = A1^T*A2 (min={C_full.min():.6f}, max={C_full.max():.6f})")
                bufC = io.BytesIO()
                figC.savefig(bufC, format='jpg', bbox_inches='tight', dpi=100)
                bufC.seek(0)
                pil_imgC = Image.open(bufC)
                plt.close(figC)

                wandb_run.report_image(
                    title=f"C = A1^T*A2 Layer {layer_idx}",
                    series="C_per_layer",
                    iteration=getattr(model, 'batch_counter', 0),
                    image=pil_imgC
                )

            # Averaging across layers
            if len(A1_mats) > 0:
                A1_avg = np.mean(np.stack(A1_mats, axis=0), axis=0)
                figA1, axA1 = plt.subplots(figsize=(8, 6))
                imA1 = axA1.imshow(A1_avg[border[0]:border[1], border[0]:border[1]], interpolation='none')
                plt.colorbar(imA1, ax=axA1)
                axA1.set_title(f"A1[avg over {len(A1_mats)} layers] (min={A1_avg.min():.6f}, max={A1_avg.max():.6f})")
                bufA1 = io.BytesIO()
                figA1.savefig(bufA1, format='jpg', bbox_inches='tight', dpi=100)
                bufA1.seek(0)
                pil_A1_avg = Image.open(bufA1)
                plt.close(figA1)
                wandb_run.report_image(
                    title="A1 Average",
                    series="A1_avg",
                    iteration=getattr(model, 'batch_counter', 0),
                    image=pil_A1_avg
                )

            if len(A2_mats) > 0:
                A2_avg = np.mean(np.stack(A2_mats, axis=0), axis=0)
                figA2, axA2 = plt.subplots(figsize=(8, 6))
                imA2 = axA2.imshow(A2_avg[border[0]:border[1], border[0]:border[1]], interpolation='none')
                plt.colorbar(imA2, ax=axA2)
                axA2.set_title(f"A2[avg over {len(A2_mats)} layers] (min={A2_avg.min():.6f}, max={A2_avg.max():.6f})")
                bufA2 = io.BytesIO()
                figA2.savefig(bufA2, format='jpg', bbox_inches='tight', dpi=100)
                bufA2.seek(0)
                pil_A2_avg = Image.open(bufA2)
                plt.close(figA2)
                wandb_run.report_image(
                    title="A2 Average",
                    series="A2_avg",
                    iteration=getattr(model, 'batch_counter', 0),
                    image=pil_A2_avg
                )

            # Compute and log averaged matrix C = A1_avg^T * A2_avg
            if len(A1_mats) > 0 and len(A2_mats) > 0:
                C_avg = A1_avg.T @ A2_avg
                figC_avg, axC_avg = plt.subplots(figsize=(8, 6))
                imC_avg = axC_avg.imshow(C_avg[border[0]:border[1], border[0]:border[1]], interpolation='none')
                plt.colorbar(imC_avg, ax=axC_avg)
                axC_avg.set_title(f"C[avg] = A1_avg^T*A2_avg (min={C_avg.min():.6f}, max={C_avg.max():.6f})")
                bufC_avg = io.BytesIO()
                figC_avg.savefig(bufC_avg, format='jpg', bbox_inches='tight', dpi=100)
                bufC_avg.seek(0)
                pil_C_avg = Image.open(bufC_avg)
                plt.close(figC_avg)
                wandb_run.report_image(
                    title="C = A1^T*A2 Average",
                    series="C_avg",
                    iteration=getattr(model, 'batch_counter', 0),
                    image=pil_C_avg
                )

    except Exception as e:
        print(f"Error in create_attention_figures (Custom_A): {e}")

    # --- 2. Attention weights of first layer ------------------------------------
    try:
        if hasattr(model, 'saved_attn') and len(model.saved_attn) > 0 and model.saved_attn[0] is not None:
            # Log across all layers and average across layers
            attn_mats = []
            for l_idx, attn_l in enumerate(model.saved_attn):
                if attn_l is None:
                    continue
                try:
                    attn_np = attn_l.detach().cpu().numpy() if torch.is_tensor(attn_l) else np.asarray(attn_l)
                except Exception:
                    # Last attempt without conversions
                    attn_np = attn_l

                crop = attn_np[border[0]:border[1], border[0]:border[1]]
                attn_mats.append(attn_np)

                fig_l, ax_l = plt.subplots(figsize=(8, 5))
                im_l = ax_l.imshow(crop, interpolation='none')
                plt.colorbar(im_l, ax=ax_l)
                ax_l.set_title(f"Attention Layer {l_idx} (avg heads) from {border[0]} to {border[1]}")
                buf_l = io.BytesIO()
                fig_l.savefig(buf_l, format='jpg', bbox_inches='tight', dpi=100)
                buf_l.seek(0)
                pil_img_layer = Image.open(buf_l)
                plt.close(fig_l)

                try:
                    wandb_run.report_image(
                        title=f"Attention Layer {l_idx}",
                        series="attention_per_layer",
                        iteration=getattr(model, 'batch_counter', 0),
                        image=pil_img_layer
                    )
                except Exception as _e:
                    print(f"Attention layer logging failed (L{l_idx}): {_e}")

            # Average across layers
            try:
                if len(attn_mats) > 0:
                    attn_avg = np.mean(np.stack(attn_mats, axis=0), axis=0)
                    crop_avg = attn_avg[border[0]:border[1], border[0]:border[1]]
                    fig_avg, ax_avg = plt.subplots(figsize=(8, 5))
                    im_avg = ax_avg.imshow(crop_avg, interpolation='none')
                    plt.colorbar(im_avg, ax=ax_avg)
                    ax_avg.set_title(f"Attention AVG over {len(attn_mats)} layers from {border[0]} to {border[1]}")
                    buf_avg = io.BytesIO()
                    fig_avg.savefig(buf_avg, format='jpg', bbox_inches='tight', dpi=100)
                    buf_avg.seek(0)
                    pil_img_avg = Image.open(buf_avg)
                    plt.close(fig_avg)

                    try:
                        wandb_run.report_image(
                            title="Attention Average",
                            series="attention_avg_layers",
                            iteration=getattr(model, 'batch_counter', 0),
                            image=pil_img_avg
                        )
                    except Exception as _e:
                        print(f"Attention avg logging failed: {_e}")
            except Exception as _e:
                print(f"Attention averaging failed: {_e}")

            # for l in range(min(2, len(model.saved_attn))):  # first two layers
            #     attn_l = model.saved_attn[l]
            #     fig3, ax3 = plt.subplots(figsize=(8, 5))
            #     im3 = ax3.imshow(attn_l[:border, :border], interpolation='none')
            #     plt.colorbar(im3, ax=ax3)
            #     ax3.set_title(f"Layer {l} Attention (avg heads) ")


                # if attn_l is not None:
                #     attn_matrices.append(attn_l.numpy())

            # if attn_matrices:
                

            #     for idx, (ax, mat) in enumerate(zip(axes, attn_matrices)):
            #         im = ax.imshow(mat[:border, :border], interpolation='none')
            #         plt.colorbar(im, ax=ax)
            #         ax.set_title(f"Layer {idx} Attention (avg heads)")
            # else:
            #     fig3 = None
        else:
            # Attention not yet computed – return None
            print("Attention not yet computed")
    except Exception as e:
        print(f"Error in create_attention_figures: {e}")

    return

def eval_buckets(model, test_dataset, train_df, itemid_col, topn=10, config=None, wandb_run=None, verbose=False):
    """Evaluate model on three popularity buckets with roughly equal numbers of interactions.

    Parameters
    ----------
    model : torch.nn.Module
        Trained SASRec model (in eval mode is not required, function will switch off grads).
    test_dataset : list
        A dataset of the form [user_train_full, user_test_list, usernum, itemnum] as expected by `evaluate`.
    train_df : pandas.DataFrame
        DataFrame containing historical interactions (used to compute popularity statistics).
    itemid_col : str
        Name of the column in `train_df` which stores item identifiers.
    topn : int, default 10
        Cut-off for ranking metrics.
    config : dict or None
        Configuration dictionary passed further to `evaluate`.
    wandb_run : wandb.sdk.wandb_run.Run or None
        Weights & Biases run for logging (optional).
    verbose : bool, default False
        If True, prints bucket scores to stdout.

    Returns
    -------
    dict
        Mapping bucket_name -> {metric: score}.
    """
    # Ensure model is in eval mode during inference
    was_training = model.training
    model.eval()

    # 1. Compute item popularity (number of interactions)
    item_counts = train_df[itemid_col].value_counts()
    item_counts = item_counts.sort_values(ascending=False)

    # --- Form buckets via cumulative sum ---
    cum = item_counts.cumsum()
    total = cum.iloc[-1]
    t1, t2 = total / 3.0, 2 * total / 3.0

    # bucket index: 0 (high), 1 (mid), 2 (low)
    bucket_idx = (cum > t2).astype(int) + (cum > t1).astype(int)
    bucket_name_map = {0: '1_high_pop', 1: '2_mid_pop', 2: '3_low_pop'}
    item_to_bucket = bucket_idx.map(bucket_name_map)

    # sets of items by buckets
    bucket_items_dict = {name: set(item_to_bucket[item_to_bucket == name].index) for name in bucket_name_map.values()}

    user_train_full, user_test_list, usernum, itemnum = test_dataset

    bucket_names = ['1_high_pop', '2_mid_pop', '3_low_pop']

    bucket_scores = {}

    for bucket_name in bucket_names:
        bucket_items = bucket_items_dict[bucket_name]
        if len(bucket_items) == 0:
            continue  # skip empty bucket
        # Filter test sequences by bucket
        filtered_test_seq = [(u, itm) for (u, itm) in user_test_list if itm in bucket_items]
        if len(filtered_test_seq) == 0:
            continue  # no pairs from this bucket in test data
        dataset_bucket = [user_train_full, filtered_test_seq, usernum, itemnum]
        results = evaluate(model, dataset_bucket, topn, config)
        scores = {metric: res["score"] for metric, res in results.items()}
        bucket_scores[bucket_name] = scores

    # Restore model's original training state
    if was_training:
        model.train()

    return bucket_scores

