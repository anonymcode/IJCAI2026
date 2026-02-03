
import gzip
import json
import os
import re
from contextlib import contextmanager, redirect_stdout
import numpy as np
import pandas as pd

# from polara.preprocessing.dataframes import reindex
# from polara.tools.display import suppress_stdout

import defaults
# from HyperbolicSASRec.lib.utils import matrix_from_observations, to_numba_dict
# from HyperbolicSASRec.lib.data import movielens as ml, amazon as amz, steam, netflix

@contextmanager
def suppress_stdout(on=True):
    if on:
        with open(os.devnull, "w") as target:
            with redirect_stdout(target):
                yield
    else:
        yield

def reindex(raw_data, index, filter_invalid=True, names=None):
    '''
    Factorizes column values based on provided pandas index. Allows resetting
    index names. Optionally drops rows with entries not present in the index.
    '''
    if isinstance(index, pd.Index):
        index = [index]

    if isinstance(names, str):
        names = [names]

    if isinstance(names, (list, tuple, pd.Index)):
        for i, name in enumerate(names):
            index[i].name = name

    new_data = raw_data.assign(**{
        idx.name: idx.get_indexer(raw_data[idx.name]) for idx in index
    })

    if filter_invalid:
        # pandas returns -1 if label is not present in the index
        # checking if -1 is present anywhere in data
        maybe_invalid = new_data.eval(
            ' or '.join([f'{idx.name} == -1' for idx in index])
        )
        if maybe_invalid.any():
            print(f'Filtered {maybe_invalid.sum()} invalid observations.')
            new_data = new_data.loc[~maybe_invalid]

    return new_data

class DataSet:
    def __init__(self, datapack, train_format=None, test_format=None, is_persistent=True, name=None):
        train, test, index = datapack
        self._index = index
        self._data_container = {
            'default': {
                'train': train, # model specific
                'test': test,   # experiment specific
            }
        }
        self.default_formats = {'train': train_format, 'test': test_format}
        self._format_kwargs = {}
        self._data_formatter = DataFormatter
        self.is_persistent = is_persistent # cache calculated data formats
        self.initialize_formats(self.default_formats)
        self.name = name if name is not None else self.__class__.__name__

    def initialize_formats(self, formats):
        for mode, format in formats.items():
            assert self.get_formatted_data(mode, format) is not None

    @property
    def user_index(self):
        return self._index['users']

    @property
    def item_index(self):
        return self._index['items']

    @property
    def train(self):
        return self.get_formatted_data('train', self.default_formats['train'])

    @property
    def test(self):
        return self.get_formatted_data('test', self.default_formats['test'])

    def get_formatted_data(self, mode, format=None):
        if format is None:
            format = 'default'
        format_kwargs = {}
        if isinstance(format, (tuple, list)):
            format, format_kwargs = format
            self._format_kwargs[format] = format_kwargs
        data_container = self._data_container.setdefault(format, {})
        try:
            formatted_data = data_container[mode]
        except KeyError: # no data in this format -> generate it from defaults
            data_formatter = self._data_formatter(format)
            formatted_data = data_formatter(self, mode, **format_kwargs)
            if self.is_persistent:
                data_container[mode] = formatted_data
        return formatted_data

    @contextmanager
    def formats(self, train=None, test=None):
        # store current values of data formats
        train_format = self.default_formats['train']
        test_format = self.default_formats['test']
        self.default_formats = { # temporarily set new data formats
            'train': train or train_format, # if not set - use current format
            'test': test or test_format # if not set - use current format
        }
        try:
            yield self
        finally: # restore initial values
            self.default_formats = {'train': train_format, 'test': test_format}

    def has_formats(self, train=None, test=None):
        train_format = train or self.default_formats['train']
        test_format = test or self.default_formats['test']

        train_format_exists = self.format_exists('train', train_format)
        test_format_exists = self.format_exists('test', test_format)
        return train_format_exists and test_format_exists

    def format_exists(self, mode, format):
        try:
            data = self._data_container[format]
        except KeyError:
            return False
        return mode in data

    def info(self):
        '''Display main dataset info and statistics'''
        logger.info(f'Dataset formats: {self.default_formats}.')
        test_data = self._data_container['default']['test']
        # TODO stepwise interactions may lack some data from default format
        userid = self.user_index.name
        itemid = self.item_index.name
        test_stats = test_data[[userid, itemid]].nunique()
        logger.info(
            f'Test data from {self.name} contains {test_data.shape[0]} interactions '
            f'between {test_stats[userid]} users and {test_stats[itemid]} items.'
        )

    def cleanup(self, formats=None):
        if formats is None:
            formats = self._data_container.keys()
            formats = list(formats)
        if not isinstance(formats, (list, set, tuple)):
            formats = [formats]
        for format in formats:
            if format != 'default': # leave source data intact
                del self._data_container[format]


class DataFormatter:
    def __init__(self, format):
        self._formatters = {
            'sparse': dataframe_to_matrix,
            'sequential': dataframe_to_sequences,
            'sequential_packed': dataframe_to_packed_sequences,
            'sequential_typed': dataframe_to_typed_sequences,
            'interactions': dataframe_to_interactions
        }
        self._target_format = format

    def __call__(self, dataset, mode, **kwargs):
        return self.format(dataset, mode, **kwargs)

    def format(self, dataset, mode, **kwargs):
        try:
            formatter = self._formatters[self._target_format]
        except KeyError:
            raise NotImplementedError(f'Unrecognized format: {self._target_format}')
        data = dataset._data_container['default'][mode]
        userid = dataset.user_index.name
        itemid = dataset.item_index.name
        timeid = defaults.timeid
        return formatter(data, userid, itemid, timeid, **kwargs)

    def register(self, format, formatter):
        self._formatters[format] = formatter


def dataframe_to_matrix(data, userid, itemid, timeid=None):
    '''Convert observations dataframe into a sparse user-item matrix.'''
    return matrix_from_observations(data, userid, itemid)


def dataframe_to_sequences(data, userid, itemid, timeid):
    '''Convert observations dataframe into a user-keyed series of lists of item sequences.'''
    return data.sort_values(timeid).groupby(userid, sort=False)[itemid].apply(list)


def dataframe_to_typed_sequences(data, userid, itemid, timeid):
    sequences = dataframe_to_sequences(data, userid, itemid, timeid)
    return to_numba_dict(sequences)


def dataframe_to_packed_sequences(data, userid, itemid, timeid):
    sequences = data.sort_values(timeid).groupby(userid, sort=True)[itemid].apply(list)
    num_users = sequences.index.max() + 1
    if len(sequences) != num_users:
        raise NotImplementedError('Only continuous `0,1,...,N-1` index of N users is supported')
    sizes = np.zeros(num_users + 1, dtype=np.intp)
    sizes[1:] = sequences.apply(len).cumsum().values
    indices = np.concatenate(sequences.to_list(), dtype=np.int32)
    return indices, sizes


def dataframe_to_interactions(
    data, userid, itemid, timeid,
    stepwise=False, max_steps=10, min_step_users=100
):
    '''Return user-item interactions either as an iterable or as a stepwise data.
    In the latter case, each step is a sequence of users with their next item.'''
    data_sorted = data[[userid, itemid, timeid]].sort_values(timeid)
    if not stepwise:
        return data_sorted[[userid, itemid]] # omit time
    return (
            data_sorted
            .assign(step = lambda df:
                df
                .groupby([userid], sort=False)[timeid]
                .transform('cumcount')
            )
            .groupby('step')[[userid, itemid]]  # omit time
            .apply(lambda x: list(x.itertuples(index=False, name=None))) # list (user,item) pairs
            .loc[lambda x: x.apply(len) >= min_step_users]
            .iloc[:max_steps] # `None` will not filter anything
            .sort_index()
        )


def get_sequence_length(dataset_name):
    dataname = re.split(r'(_\d+$)', dataset_name)[0] # remove pcore setting if present
    if dataname in (ml.DATASETS | netflix.DATASETS):
        return defaults.sequence_length_movies
    elif (dataname.lower() in amz.ALIAS) or (dataname in amz.DATASETS):
        return defaults.sequence_length_amazon
    elif dataname in steam.DATASETS:
        return defaults.sequence_length_amazon
    raise ValueError(f'Unrecognized dataset {dataname}')


def prepare_data(dataset_name, time_offset_q=None):
    # print("defaults.max_test_interactions", defaults.max_test_interactions)
    time_offset_valid, time_offset_test = read_time_offsets(time_offset_q)
    userid, itemid, timeid = "userid", "itemid", "timestamp"
    data = read_raw_data(dataset_name)
    
    if time_offset_valid is None:
        train_data_, test_data_ = split_data_by_time(
            data, time_offset_test, timeid, max_samples=defaults.max_test_interactions
        ) # split by time 0.95 example or 50k interactions
        test_datapack = reindex_data(train_data_.copy(), test_data_, userid, itemid)
        return (test_datapack,)

    eval_offset = time_offset_test + time_offset_valid - 1
    valid_ratio = (1 - time_offset_valid) / (1 - eval_offset)
    max_valid_ratio = defaults.max_test_interactions / (len(data) * (1 - eval_offset))
    if valid_ratio > max_valid_ratio: # extend the offset to preserve valid/test data size ratio
        eval_offset = 1 - defaults.max_test_interactions / (len(data) * valid_ratio)

    train_data_valid_, rest_data_ = split_data_by_time(data, eval_offset, timeid)
    valid_data_, test_data_ = split_data_by_time(rest_data_, valid_ratio, timeid)

    tune_datapack = reindex_data(
        train_data_valid_.copy(), valid_data_, userid, itemid
    )
    train_data_ = pd.concat([train_data_valid_, valid_data_], axis=0)
    
    test_datapack = reindex_data(
        train_data_.copy(), test_data_, userid, itemid
    )
    return tune_datapack, test_datapack


def prepare_data_exclude(dataset_name, time_offset_q=None):
    """An extended version of :func:`prepare_data` that *excludes* any validation
    or test interactions whose item ids are absent from the *training* split.

    This is useful when the recommendation model is trained **only** on the
    training data and its item-embedding matrix must therefore contain
    **all** items that will appear later during validation and test phases.

    The splitting logic (temporal split with optional `time_offset_q`)
    remains exactly the same as in :func:`prepare_data`. The only difference
    is that the item vocabulary is *frozen* after the first split and the
    subsequent validation/test parts are re-indexed against this fixed
    vocabulary with ``filter_invalid=True`` so that previously unseen items
    are dropped.

    Returns
    -------
    tuple
        Same structure as :func:`prepare_data`: ``(tune_datapack, test_datapack)``
        where each element is a triple ``(train_df, test_df, data_index)``.
    """
    # ----------------------- Step 1: read & split raw data --------------------
    time_offset_valid, time_offset_test = read_time_offsets(time_offset_q)
    userid, itemid, timeid = "userid", "itemid", "timestamp"
    data = read_raw_data(dataset_name)

    # ---------- Helper to build datapack with *fixed* item vocabulary ----------
    def _reindex_against_fixed_vocab(train_df_fixed, df_to_reindex, idx_fixed):
        """Re-index *df_to_reindex* using the *fixed* item vocabulary contained in
        *idx_fixed* (pandas.Index). Items outside the vocabulary are dropped.
        Users are re-indexed in the same way as :func:`reindex_data`.
        """
        # Re-index items first (drop cold-start items)
        with suppress_stdout(True):
            df_part = reindex(df_to_reindex.copy(), idx_fixed['items'], filter_invalid=True)

        # Re-index users (allowing new users to appear)
        test_user_idx = idx_fixed['users'].get_indexer(df_part[userid])
        is_new_user = test_user_idx == -1
        if is_new_user.any():
            new_user_idx, idx_fixed['new_users'] = pd.factorize(df_part.loc[is_new_user, userid])
            test_user_idx[is_new_user] = new_user_idx + len(idx_fixed['users'])
        df_part.loc[:, userid] = test_user_idx
        return df_part

    # ------------------------- Only train/test split --------------------------
    if time_offset_valid is None:
        train_data_raw, test_data_raw = split_data_by_time(
            data, time_offset_test, timeid, max_samples=defaults.max_test_interactions
        )
        # First re-index train (creates the fixed item vocab)
        train_df, _, data_index = reindex_data(train_data_raw.copy(), pd.DataFrame([], columns=data.columns), userid, itemid)
        # Now re-index test against the *fixed* vocabulary
        test_df = _reindex_against_fixed_vocab(train_df, test_data_raw, data_index)
        test_datapack = (train_df, test_df, data_index)
        return (test_datapack,)

    # ---------------------------- Train/Valid/Test ----------------------------
    eval_offset = time_offset_test + time_offset_valid - 1
    valid_ratio = (1 - time_offset_valid) / (1 - eval_offset)
    max_valid_ratio = defaults.max_test_interactions / (len(data) * (1 - eval_offset))
    if valid_ratio > max_valid_ratio:
        eval_offset = 1 - defaults.max_test_interactions / (len(data) * valid_ratio)

    train_data_valid_raw, rest_data_raw = split_data_by_time(data, eval_offset, timeid)
    valid_data_raw, test_data_raw = split_data_by_time(rest_data_raw, valid_ratio, timeid)

    # --- 1. Re-index Train & Valid (creates our *fixed* vocabulary) -----------
    train_df, valid_df, data_index = reindex_data(
        train_data_valid_raw.copy(), valid_data_raw, userid, itemid
    )
    # valid_df already contains only items from *train_df* at this point.
    tune_datapack = (train_df, valid_df, data_index)

    # --- 2. Build Test datapack using the SAME vocabulary --------------------
    # Combine *filtered* train & valid as the historical context for test phase
    train_full_df = pd.concat([train_df, valid_df], axis=0)

    # Re-index the *raw* test data against the *fixed* item vocabulary
    test_df = _reindex_against_fixed_vocab(train_df, test_data_raw, data_index)

    test_datapack = (train_full_df, test_df, data_index)
    return tune_datapack, test_datapack


def prepare_data_cut(dataset_name, time_offset_q=None):
    """Variant of :func:`prepare_data` that *cuts* validation and test
    sequences at the first appearance of an item **not** present in the
    training split.

    For each user we iterate over interactions in chronological order; once an
    unseen item is encountered, this interaction **and all subsequent ones**
    are removed. This guarantees that every remaining validation/test target
    item exists in the item vocabulary learned from training data, while
    preserving as much temporal context as possible.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (file ``ScalableSASRec/preprocessed_data/{name}.csv`` must exist).
    time_offset_q : float or (float, float) or None
        Same semantics as in :func:`prepare_data` – quantiles for temporal
        split into validation/test (single value means *only* test split).

    Returns
    -------
    tuple
        ``(tune_datapack, test_datapack)`` identical in structure to the
        output of :func:`prepare_data`/`prepare_data_exclude`.
    """
    # ------------------------------------------------------------------
    time_offset_valid, time_offset_test = read_time_offsets(time_offset_q)
    userid, itemid, timeid = "userid", "itemid", "timestamp"
    data = read_raw_data(dataset_name)

    # ---------------- helper: cut by unseen items ----------------------
    def _cut_sequences(df, vocab_items):
        """Return a tuple (df_cut, users_with_cold).

        df_cut – DataFrame where each user's sequence is truncated **before** the
        first unseen item (as before).
        users_with_cold – set(uid) for which a cold item occurred *anywhere* in
        their original sequence (i.e. the cut actually happened). This allows
        us to exclude those users from later splits, as requested.
        """
        if df.empty:
            return df, set()  # quick exit

        df_sorted = df.sort_values(timeid)
        parts = []
        users_with_cold = set()

        for uid, grp in df_sorted.groupby(userid, sort=False):
            items = grp[itemid].tolist()
            cut_pos = len(items)  # by default keep all
            for idx_i, itm in enumerate(items):
                if itm not in vocab_items:
                    cut_pos = idx_i
                    users_with_cold.add(uid)  # remember this uid
                    break
            if cut_pos > 0:
                parts.append(grp.iloc[:cut_pos])

        if parts:
            return pd.concat(parts, axis=0), users_with_cold
        # every sequence started with a cold item
        return df.iloc[0:0], users_with_cold

    # ---------------- helper: re-index with a *fixed* vocabulary --------
    def _reindex_against_fixed_vocab(df_to_reindex, idx_fixed):
        """Re-index *df_to_reindex* using *idx_fixed['items']*; drop cold items."""
        with suppress_stdout(True):
            df_part = reindex(df_to_reindex.copy(), idx_fixed['items'], filter_invalid=True)
        # users
        test_user_idx = idx_fixed['users'].get_indexer(df_part[userid])
        is_new_user = test_user_idx == -1
        if is_new_user.any():
            new_user_idx, idx_fixed['new_users'] = pd.factorize(df_part.loc[is_new_user, userid])
            test_user_idx[is_new_user] = new_user_idx + len(idx_fixed['users'])
        df_part.loc[:, userid] = test_user_idx
        return df_part

    # ---------------- Case 1: only Train/Test --------------------------
    if time_offset_valid is None:
        train_raw, test_raw = split_data_by_time(
            data, time_offset_test, timeid, max_samples=defaults.max_test_interactions
        )
        train_vocab = set(train_raw[itemid].unique())
        test_cut_raw, users_with_cold = _cut_sequences(test_raw, train_vocab)

        # re-index
        train_df, _, data_index = reindex_data(train_raw.copy(), pd.DataFrame([], columns=data.columns), userid, itemid)
        test_df = _reindex_against_fixed_vocab(test_cut_raw, data_index)
        test_datapack = (train_df, test_df, data_index)
        return (test_datapack,)

    # ---------------- Case 2: Train / Valid / Test ---------------------
    eval_offset = time_offset_test + time_offset_valid - 1
    valid_ratio = (1 - time_offset_valid) / (1 - eval_offset)
    max_valid_ratio = defaults.max_test_interactions / (len(data) * (1 - eval_offset))
    if valid_ratio > max_valid_ratio:  # extend offset to keep valid/test size ratio
        eval_offset = 1 - defaults.max_test_interactions / (len(data) * valid_ratio)

    train_raw, rest_raw = split_data_by_time(data, eval_offset, timeid)
    valid_raw, test_raw = split_data_by_time(rest_raw, valid_ratio, timeid)

    # ---- 1) Training & Validation -------------------------------------
    train_vocab = set(train_raw[itemid].unique())
    valid_cut_raw, users_with_cold = _cut_sequences(valid_raw, train_vocab)

    train_df, valid_df, data_index = reindex_data(
        train_raw.copy(), valid_cut_raw, userid, itemid
    )
    tune_datapack = (train_df, valid_df, data_index)

    # ---- 2) Test -------------------------------------------------------
    # Remove from test_raw users who had cold items in validation
    test_raw_filtered = test_raw.loc[~test_raw[userid].isin(users_with_cold)].copy()

    train_full_raw = pd.concat([train_raw, valid_cut_raw], axis=0)
    train_full_vocab = set(train_full_raw[itemid].unique())
    test_cut_raw, _ = _cut_sequences(test_raw_filtered, train_full_vocab)

    test_df = _reindex_against_fixed_vocab(test_cut_raw, data_index)
    train_full_df = pd.concat([train_df, valid_df], axis=0)
    test_datapack = (train_full_df, test_df, data_index)

    return tune_datapack, test_datapack


def read_time_offsets(time_offset_q):
    '''
    Always returns (validation, test) offsets tuple.
    '''
    if isinstance(time_offset_q, (list, tuple)):
        time_offset_valid, time_offset_test = time_offset_q
        return time_offset_valid, time_offset_test
    return None, time_offset_q or defaults.time_offset_q


def read_raw_data(dataset_name):
    data_path = os.path.join(defaults.data_dir, f'{dataset_name}.csv')
    return pd.read_csv(data_path, na_filter=False)  # assume data is clean (mostly for steam data)

    # return pd.read_csv(f'preprocessed_data/{dataset_name}.csv', na_filter=False)  # assume data is clean (mostly for steam data)

    # return pd.read_csv(f'preprocessed_data/{dataset_name}.csv', na_filter=False)  # assume data is clean (mostly for steam data)


# def entity_names(dataset_name):
#     userid = defaults.userid
#     timeid = defaults.timeid
#     dataname = re.split(r'(_\d+$)', dataset_name)[0] # remove pcore setting if present
#     if dataname in (ml.DATASETS | netflix.DATASETS):
#         itemid = 'movieid'
#     elif (dataname.lower() in amz.ALIAS) or (dataname in amz.DATASETS):
#         itemid = 'asin'
#     elif dataname in steam.DATASETS:
#         itemid = 'product_id'
#     else:
#         raise ValueError(f'Unrecognized dataset {dataset_name}')
#     return userid, itemid, timeid


def split_data_by_time(data, time_q, timeid, max_samples=None):
    test_timepoint = data[timeid].quantile(q=time_q, interpolation='nearest')
    test_time = data[timeid] >= test_timepoint
    test_data = data.loc[test_time, :]
    if max_samples is not None and len(test_data) > max_samples:
        # If the number of rows in test_data exceeds max_samples,
        # take only the latest rows based on timeid
        test_data = test_data.sort_values(by=timeid, ascending=True).tail(max_samples)
        train_data = data.drop(test_data.index)
    else:
        train_data = data.loc[~test_time, :]
    return train_data, test_data


def reindex_data(train, test, userid, itemid, verbose=False):
    train_data, data_index = transform_indices(train, userid, itemid)
    # reindex items (and exclude cold-start items)
    with suppress_stdout(not verbose): # do not print how many entries were filtered
        test_data = reindex(test, data_index['items'], filter_invalid=True)
    # reindex users
    test_user_idx = data_index['users'].get_indexer(test_data[userid])
    is_new_user = test_user_idx == -1
    if is_new_user.any(): # track unseen users - to be used in warm-start regime
        new_user_idx, data_index['new_users'] = pd.factorize(test_data.loc[is_new_user, userid])
        # ensure no intersection with train users index
        test_user_idx[is_new_user] = new_user_idx + len(data_index['users'])
    # assign new user index
    test_data.loc[:, userid] = test_user_idx
    return train_data, test_data, data_index


def transform_indices(data, users, items):
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        codes, index = to_numeric_id(data, field)
        data_index[entity] = index
        data.loc[:, field] = codes
    return data, data_index


def to_numeric_id(data, field):
    idx_data = data[field].astype("category")
    codes = idx_data.cat.codes
    index = idx_data.cat.categories.rename(field)
    return codes, index