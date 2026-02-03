import os
import sys
import re
# import argparse
# import importlib
# import shutil
from collections import defaultdict
# from contextlib import contextmanager
import gzip
import json
import urllib
import numpy as np
import pandas as pd
from multiprocessing import Process, Queue
# from polara import get_movielens_data
from ast import literal_eval

# from trash.get_new_dataset import get_movielens_10_data

try:
    import wandb
except ImportError:
    wandb = None

import torch
from torch.utils.data import Dataset, DataLoader

###################################

##### from SATF.utils import data_partition #####

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    with open('data/%s.txt' % fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        user_items = User[user]
        nfeedback = len(user_items)
        if nfeedback < 3:
            user_train[user] = user_items
        else:
            user_train[user] = user_items[:-2]
            user_valid[user] = user_items[-2]
            user_test[user] = user_items[-1]
    return [user_train, user_valid, user_test, usernum, itemnum]
###########

##### from SATF.preprocessing import split_offsets, generate_partition #####

def is_sequential(left, right, userid, timeid):
    r_time = right.groupby(userid)[timeid].min()
    l_time = left.query(f'{userid} in @r_time.index').groupby(userid)[timeid].max()
    # assume r_time contains unseen users
    maybe_seq = l_time.combine(r_time, lambda x, y: x <= y)
    result = maybe_seq.all()
    if not result: # maybe it's due to unseen users
        maybe_unseen = maybe_seq[~maybe_seq].index # potensially unseen users idx
        if maybe_unseen.isin(l_time).any(): # they are not unseen => contradiction
            return False
        # check that unseen users all go later
        result = r_time.loc[maybe_unseen].min() >= l_time.max() 
    # handle remaining users not present in the next step
    rest = left.query(f'{userid} not in @r_time.index')
    if len(rest):
        result = result and (rest[timeid].max() <= r_time.min())
    return result

def reindex_data(data, columns, base=1):
    '''Reindex starting from `base`. SASRec relies on indexing that starts from 1.'''
    # Fix random seed for deterministic pandas operations
    np.random.seed(42)
    
    categories = {}
    new_index = {}
    
    for col in columns:
        # Create categories with deterministic order (sorting)
        unique_values = sorted(data[col].unique())  # Sort for determinism
        cat = pd.Categorical(data[col], categories=unique_values)
        categories[col] = cat
        new_index[col] = pd.Index(np.r_[range(-base, 0), unique_values])
    
    new_data = data.assign(**{col: cat.codes+base for col, cat in categories.items()})
    return new_data, new_index

def verify_reindex(data, userid, itemid, data_index):
    new_data = (
        data
        .assign(**{col: idx.get_indexer_for(data[col]) for col, idx in data_index.items()})
    )    
    warm_users = (new_data[userid] == -1) & (new_data[itemid] != -1)
    if warm_users.any(): # allow unseen users with known items, update corresponding user index
        # do not explicitly assume that users are unique in the dataset
        new_user_cat = data.loc[warm_users, userid].astype('category').cat
        # assign new user index
        base = len(data_index[userid])
        new_data.loc[warm_users, userid] = base + new_user_cat.codes.astype('int64')
        # update user index data
        data_index[userid] = data_index[userid].append(pd.Index(new_user_cat.categories))
    return new_data.query(f'{itemid} != -1') # skip unseen items

def read_offsets(offset_str):
    offsets = re.findall(r'(\d+)(\w+)', offset_str)
    return [pd.DateOffset(**{interval: int(n_intervals)}) for n_intervals, interval in offsets]

def split_offsets(data, offset_str, timeid, time_unit='s'):
    '''Data reading and proper indexing. Adapted for evaluation of SASRec that relies on 1-based indexing'''
    # Fix random seed for pandas
    np.random.seed(42)  # use fixed seed for pandas operations
    
    time_offsets = read_offsets(offset_str) # [DateOffset(), DateOffset()]
    timestamps = pd.to_datetime(data[timeid], unit=time_unit)
    
    test_time_threshold = timestamps.max() - time_offsets.pop() # we use the last time_offset for test

    valid_time_threshold = test_time_threshold - time_offsets.pop() # we use the last time_offset without test  for valid
    
    train_split = timestamps <= valid_time_threshold
    valid_split = (~train_split) & (timestamps <= test_time_threshold)
    test_split = timestamps > test_time_threshold

    train = data.loc[train_split] # dataframes 
    valid = data.loc[valid_split]
    test = data.loc[test_split]

    # print("train", train)
    # print("valid", valid)
    # print("test", test)

    return train, valid, test

def generate_partition(observed: pd.DataFrame, holdout: pd.DataFrame, userid: str, itemid: str, timeid: str, stepwise_eval: bool):
    if observed is None:
        return
    
    idx_start = 1 # to conform with SASRec indexing
    train, data_index = reindex_data(observed, [userid, itemid], base=idx_start)
    
    usernum = len(data_index[userid]) - idx_start
    itemnum = len(data_index[itemid]) - idx_start

    user_train = (
        train
        .sort_values(timeid)
        .groupby(userid, sort=False)[itemid]
        .apply(list)
        .to_dict()
    )

    holdout = verify_reindex(holdout, userid, itemid, data_index)
    if stepwise_eval:
        user_test = generate_sequences(holdout, userid, itemid, timeid, stepwise=stepwise_eval)
    else:
        data_sorted = holdout.sort_values(timeid)
        user_test = (
            data_sorted
            .sort_values(timeid)
            .groupby(userid, sort=False)[itemid]
            .apply(list)
            .to_dict()
        )   

    return user_train, user_test, usernum, itemnum

def generate_sequences(data, userid, itemid, timeid, stepwise, max_steps=10, min_step_users=100):
    '''Each step is a sequence of users with their next item.'''
    
    data_sorted = data.sort_values(timeid)
    if stepwise:
        return (
            data_sorted
            .assign(step = lambda df:
                df
                .groupby([userid], sort=False)[timeid]
                .transform('cumcount')
            )
            .groupby('step')[[userid, itemid]]
            .apply(lambda x: list(x.itertuples(index=False, name=None))) # list (user,item) pairs
            .loc[lambda x: x.apply(len) >= min_step_users]
            .iloc[:max_steps] # `None` will not filter anything
            .sort_index()
        )
    return list(data_sorted[[userid, itemid]].itertuples(index=False, name=None))
##########

##### from SATF.utils import WarpSampler, get_wandb, join_str #####

# sampler for batch generation
def random_neq(l, r, s, random_state):
    t = random_state.randint(l, r)
    while t in s:
        t = random_state.randint(l, r)
    return t

class SASRecDataset(Dataset):
    """
    PyTorch Dataset for SASRec with deterministic batch generation
    """
    def __init__(self, user_train, usernum, itemnum, maxlen, seed=42):
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.seed = seed
        
        # Create list of valid users (those with more than 1 item)
        self.valid_users = [user for user in range(1, usernum + 1) 
                           if len(user_train.get(user, [])) > 1]
        
        # Use fixed dataset size for determinism
        self.dataset_size = len(self.valid_users) * 10  # each user can provide multiple samples
        
        # Create deterministic random state
        self.random_state = np.random.RandomState(seed)
        
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        # Use idx for deterministic user selection
        user_idx = idx % len(self.valid_users)
        user = self.valid_users[user_idx]
        
        # Create local random state based on idx for determinism
        local_random_state = np.random.RandomState(self.seed + idx)
        
        # Get user items (they should be valid by definition)
        user_items = self.user_train[user]
        
        # Generate sequences
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        
        nxt = user_items[-1]
        idx_seq = self.maxlen - 1
        
        ts = set(user_items)
        for i in reversed(user_items[:-1]):
            seq[idx_seq] = i
            pos[idx_seq] = nxt
            neg[idx_seq] = random_neq(1, self.itemnum + 1, ts, local_random_state)
            nxt = i
            idx_seq -= 1
            if idx_seq == -1:
                break
        
        return user, seq, pos, neg

def worker_init_fn(worker_id):
    """
    Initialization function for DataLoader workers
    Ensures determinism in multiprocessing environment
    """
    # Get base seed from torch
    base_seed = torch.initial_seed() % (2**32)
    # Create unique seed for each worker
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)

class TorchSampler:
    """
    Replacement for WarpSampler based on PyTorch DataLoader
    """
    def __init__(self, user_train, usernum, itemnum, batch_size=64, maxlen=200, n_workers=1, seed=42):
        self.batch_size = batch_size
        self.seed = seed
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.n_workers = n_workers
        
        # Create fixed generator for determinism
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        
        # Counter to track number of created DataLoaders
        self.dataloader_count = 0
        
        # Create dataset
        self.dataset = SASRecDataset(
            user_train=user_train,
            usernum=usernum, 
            itemnum=itemnum,
            maxlen=maxlen,
            seed=seed
        )
        
        # Create DataLoader with deterministic settings
        self._create_dataloader()
        
    def _create_dataloader(self):
        """Creates a new DataLoader while preserving generator state"""
        # Create new generator with deterministic seed
        # Use dataloader_count to ensure uniqueness but determinism
        new_generator = torch.Generator()
        new_generator.manual_seed(self.seed + self.dataloader_count)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,  # shuffle enabled, but with fixed generator for determinism
            num_workers=self.n_workers,
            pin_memory=True,
            drop_last=True,  # important for batch size stability
            worker_init_fn=worker_init_fn,
            generator=new_generator,  # use new generator with deterministic seed
            persistent_workers=True if self.n_workers > 0 else False
        )
        
        # Create iterator
        self.iterator = iter(self.dataloader)
        self.dataloader_count += 1
        
    def next_batch(self):
        """
        Get next batch
        """
        try:
            batch = next(self.iterator)
            # Convert to required format (as in original WarpSampler)
            users, seqs, pos, neg = batch
            return (
                users.numpy(),
                seqs.numpy(), 
                pos.numpy(),
                neg.numpy()
            )
        except StopIteration:
            # If iterator is exhausted, create new one with deterministic seed
            self._create_dataloader()
            batch = next(self.iterator)
            users, seqs, pos, neg = batch
            return (
                users.numpy(),
                seqs.numpy(),
                pos.numpy(), 
                neg.numpy()
            )
    
    def close(self):
        """
        Close sampler (for compatibility with original API)
        """
        # DataLoader automatically manages resources
        pass

# For backward compatibility create alias
WarpSampler = TorchSampler

class DummyWandb:
    def init(self, **kwargs):
        return self
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

def get_wandb(dummy=False):
    # if dummy or (wandb is None):
    #     return DummyWandb()
    return wandb

def join_str(*args):
    return '_'.join(filter(None, args))

####



def check_dir(DATA_DIR):
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)   

##############
def ml_10_downloader(DATA_DIR):
    data = get_movielens_10_data(include_time=True)
    dest = os.path.join(DATA_DIR, 'ml-10m.gz')
    check_dir(DATA_DIR)
    (
        data
        .loc[:, ['userid', 'movieid', 'timestamp']]
        .to_csv(dest, index=False)
    )
    print(f'ML-10M data saved to {dest}.')

##############

def ml_downloader(DATA_DIR):
    data = get_movielens_data(include_time=True)
    dest = os.path.join(DATA_DIR, 'ml-1m.gz')
    check_dir(DATA_DIR)
    (
        data
        .loc[:, ['userid', 'movieid', 'timestamp']]
        .to_csv(dest, index=False)
    )
    print(f'ML-1M data saved to {dest}.')


def parse_lines_amz(path, fields):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            yield json.loads(line, object_hook=lambda dct: tuple(dct[key] for key in fields))

def amz_downloader(DATA_DIR, dataset):
    dsname = {
        'amz-b': 'Beauty',
        'amz-g': 'Toys_and_Games',
    }
    name = dsname[dataset]
    url = f'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{name}_5.json.gz'
    tmp_file, _ = urllib.request.urlretrieve(url) # this may take some time depending on your internet connection
    print(f'Saved temporary file to {tmp_file}. Processing...')
    fields = ['reviewerID', 'asin', 'unixReviewTime']
    data = pd.DataFrame.from_records(parse_lines_amz(tmp_file, fields), columns=fields)
    dest = os.path.join(DATA_DIR, f'{dataset}.gz')
    check_dir(DATA_DIR)
    (
        data
        .rename(columns={'reviewerID': 'userid', 'unixReviewTime': 'timestamp'})
        .to_csv(dest, index=False)
    )
    print(f'{dataset.upper()} data saved to {dest}.')


def parse_lines_steam(path, fields):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            dct = literal_eval(line.strip())
            yield {key: dct[key] for key in fields}

def pcore_filter(data, pcore, userid, itemid):
    while pcore: # do only if pcore is specified
        item_check = True
        valid_items = data[itemid].value_counts() >= pcore
        if not valid_items.all():
            data = data.query(
                f'{itemid} in @valid_items.index[@valid_items]'
            )
            item_check = False
            
        user_check = True
        valid_users = data[userid].value_counts() >= pcore
        if not valid_users.all():
            data = data.query(
                f'{userid} in @valid_users.index[@valid_users]'
            )
            user_check = False
        
        if user_check and item_check:
            break
    return data.copy()

def steam_downloader(DATA_DIR):
    url = f'http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz'
    #did not pass sertificate verification for some reason
    tmp_file, _ = urllib.request.urlretrieve(url) # this may take some time depending on your internet connection
    
###################################################
# #     with open(file_name, 'wb') as f:
# #     tmp_file = requests.get(url, verify=False)
# #     f.write(resp.content)


#     ctx = ssl.create_default_context()
#     ctx.check_hostname = False
#     ctx.verify_mode = ssl.CERT_NONE

#     tmp_file, _ = urllib.request.urlretrieve(url, context=ctx)
###################################################    
    
    print(f'Saved temporary file to {tmp_file}. Processing...')
    fields = ['username', 'product_id', 'date']
    raw_data = pd.DataFrame.from_records(parse_lines_steam(tmp_file, fields), columns=fields)
    data_dedup = raw_data.drop_duplicates(subset=['username', 'product_id'], keep='last')
    data_clean = pcore_filter(data_dedup, 5, 'username', 'product_id')

    data_clean.loc[:, 'timestamp'] = (
        pd.to_datetime(data_clean['date']) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta('1s')
    dest = os.path.join(DATA_DIR, 'steam.gz')
    check_dir(DATA_DIR)
    (
        data_clean
        .loc[:, ['username', 'product_id', 'timestamp']]
        .rename(columns={'username': 'userid'})
        .to_csv(dest, index=False)
    )
    print(f'Steam data saved to {dest}.')


def download_all_data(DATA_DIR):
    ml_downloader(DATA_DIR)
    amz_downloader('amz-b')
    amz_downloader('amz-g')
    steam_downloader(DATA_DIR)


    
################################################
def steam_preproc(DATA_DIR):
    url = f'http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz'
    #did not pass sertificate verification for some reason
    tmp_file, _ = urllib.request.urlretrieve(url) # this may take some time depending on your internet connection
    
###################################################
# #     with open(file_name, 'wb') as f:
# #     tmp_file = requests.get(url, verify=False)
# #     f.write(resp.content)


#     ctx = ssl.create_default_context()
#     ctx.check_hostname = False
#     ctx.verify_mode = ssl.CERT_NONE

#     tmp_file, _ = urllib.request.urlretrieve(url, context=ctx)
###################################################    
    
    print(f'Saved temporary file to {tmp_file}. Processing...')
    fields = ['username', 'product_id', 'date']
    raw_data = pd.DataFrame.from_records(parse_lines_steam(tmp_file, fields), columns=fields)
    data_dedup = raw_data.drop_duplicates(subset=['username', 'product_id'], keep='last')
    data_clean = pcore_filter(data_dedup, 5, 'username', 'product_id')

    data_clean.loc[:, 'timestamp'] = (
        pd.to_datetime(data_clean['date']) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta('1s')
    dest = os.path.join(DATA_DIR, 'steam.gz')
    check_dir(DATA_DIR)
    (
        data_clean
        .loc[:, ['username', 'product_id', 'timestamp']]
        .rename(columns={'username': 'userid'})
        .to_csv(dest, index=False)
    )
    print(f'Steam data saved to {dest}.')



def generate_sequential(train_data, userid, itemid, maxlen, seqid='position'):
    return (
        pd.concat(
            {
                user: pd.DataFrame(
                    data = {itemid: items},
                    index = range(maxlen-len(items), maxlen)
                )
                for user, items in train_data.items()
            },
            names=[userid, seqid]
        )
        .sort_index()
        .reset_index()
    )

def entity_names(dataset_name):
    timeid = 'timestamp'
    userid = 'userid'
    if dataset_name.lower().startswith('ml-'):
        itemid = 'movieid'
    elif dataset_name.lower().startswith('amz'):
        itemid = 'asin'
    elif dataset_name.lower().startswith('steam'):
        itemid = 'product_id'
    elif dataset_name.lower().startswith('zvuk'):
        itemid = 'trackid'        
    else:
        raise ValueError('Unrecognized dataset')
    return userid, itemid, timeid

def data_to_df(dataset, userid, itemid, timeid):
    *data, usernum, itemnum = dataset
    def convert_data(dat):
        return (
            pd.Series(dat, name=itemid)
            .rename_axis(userid)
            .explode() # ravel item lists if present
            .reset_index()
            .assign(**{f'{timeid}': lambda x: range(len(x))})
        )
    return [convert_data(d) for d in data] + [usernum, itemnum]

def read_dataset(dataset_name, offset_str, stepwise_eval=False, part='all', verbose=False):
    is_validation = part.lower() == 'valid'
    read_all = part.lower() == 'all'
    dataset_valid = dataset_test = None
    # read data
    if offset_str: # use Global time split in format '4months-4months' to split data
        data = pd.read_csv(f'./data/{dataset_name}.gz')
        userid, itemid, timeid = entities = entity_names(dataset_name)
        train, valid, test = split_offsets(data, offset_str, timeid)
        # print("do we split this ")
    else: # or use ready leave-one-out split
        user_train, user_valid, user_test, *stats = data_partition(dataset_name)
        # print("or this?")


    # prepare validation part
    if is_validation or read_all:
        if offset_str:
            dataset_valid = generate_partition(train, valid, *entities, stepwise_eval=stepwise_eval)
            print('do we visit this shit')
        else:
            dataset_valid = [user_train, user_valid] + stats
            print('or this shit')

    
    # prepare test part
    if not is_validation:
        if offset_str:
            print("last but not least")
            # train_full = train.append(valid)
            train_full = pd.concat([train, valid], ignore_index=True)

            dataset_test = generate_partition(train_full, test, *entities, stepwise_eval=stepwise_eval)
            # dataset_test = [train_full_dict, test_list, usernum, itemnum]
            
        else:
            user_train_full = {user: user_train.get(user, []) + [item] for user, item in user_valid.items()}
            dataset_test = [user_train_full, user_test] + stats
            print("last and least")
        # dataset_valid it is train -> val, dataset_test it is train + val -> test
    return dataset_valid, dataset_test # 


def sequential_training_data(user_train, userid, itemid, maxlen, seqid='position'):
    train_data = generate_sequential(user_train, userid, itemid, maxlen, seqid=seqid)
    recdata = RecommenderData(train_data, userid, itemid, feedback=seqid)
    recdata.verbose = False
    recdata.prepare_training_only()
    return recdata


def display_stats(dataset):
    [observed, holdout, usernum, itemnum] = dataset
    if isinstance(holdout, dict):
        num_test_users = {0: len(holdout)}
        print(f'# of test users per step: {num_test_users}')
    elif isinstance(holdout, list):
        num_test_events = len(holdout)
        print(f'# of test events: {num_test_events}')
    else:
        num_test_users = holdout.apply(len).to_dict()
        print(f'# of test users per step:\n{num_test_users}')
################