import os
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import sparse as sp
import torch
from prefetch_generator import BackgroundGenerator

from utils.matrix import biadj_to_laplacian, biadj_to_propagation, sparse_mx_to_torch_sparse_tensor

step_name = ['train', 'valid', 'test']
step_counter = 0


class ISRDataset:
    """ Incremental sequential recommendation dataset """
    def __init__(self, args, df, df_0=None, mask_trend=None):
        self.n_user = args.n_user
        self.n_item = args.n_item
        self.n_neg_sample = args.n_neg_sample
        self.ts_max = args.ts_max
        self.idx_pad = args.idx_pad

        self.all_i = pd.Series(range(1, self.n_item))  # remove padding 0
        self.all_u = pd.Series(range(1, self.n_user))
        self.df = df

        # get index of unique timestamps
        self.ts_unique = np.unique(df['t'])
        self.n_ts = len(self.ts_unique)
        end_idx_ts_dict = {t: i + 1 for i, t in enumerate(df['t'])}

        # inherit previous interactions
        self.t0 = self.df['t'].min()
        if df_0 is not None:
            len_df_0 = df_0.shape[0]
            self.cum_n_records = np.array([len_df_0] + [end_idx_ts_dict[t] + len_df_0 for t in self.ts_unique])
            self.df = pd.concat([df_0, self.df])
        else:
            self.cum_n_records = np.array([0] + [end_idx_ts_dict[t] for t in self.ts_unique])
            self.df = df

        # trend file
        self.cal_trend = args.cal_trend
        if args.cal_trend:
            if mask_trend is not None:
                self.mask_trend = mask_trend
            else:
                self.gen_trend(args.k_trend, args.len_trend)

            self.n_trend = self.mask_trend[:, 0, :].sum()
            self.n_non_trend = self.mask_trend[:, 1, :].sum()
        else:
            self.mask_trend = None
            self.n_trend = self.n_non_trend = 1

    def gen_trend(self, k_trend, len_trend):
        self.mask_trend = torch.zeros(self.n_ts, 2, self.n_item - 1)

        global step_counter
        for idx in tqdm(range(self.n_ts), desc=f'- getting {step_name[step_counter]} trend', leave=False):
            ts_now = self.ts_unique[idx]
            df_tmp = self.df.loc[(self.df['t'] < ts_now) & (self.df['t'] >= (ts_now - len_trend))]
            if len(df_tmp) > 0:
                occur_i = df_tmp['i'].to_list()
                for i in df_tmp['i'].unique():
                    if occur_i.count(i) >= k_trend:
                        self.mask_trend[idx, 0, i - 1] = 1
                    else:
                        self.mask_trend[idx, 1, i - 1] = 1
        step_counter += 1

    def __len__(self):
        return len(self.ts_unique)

    def __getitem__(self, idx):
        return self.getitem(idx)

    def getitem(self, idx, output_diff=True):
        ts_now = self.ts_unique[idx]
        t_diff = (ts_now - (self.ts_unique[idx - 1] if idx > 0 else self.t0)) / self.ts_max

        idx_tgt_start = self.cum_n_records[idx]
        idx_tgt_end = self.cum_n_records[idx + 1]
        df_obs_tgt = self.df.iloc[:idx_tgt_end]  # all happened interactions

        # get instant interactions graph (both positive and negative targets)
        df_tgt = self.df.iloc[idx_tgt_start:idx_tgt_end]
        adj_ins = self.build_ui_mat(df_tgt)
        tgt_u, tgt_i = df_tgt['u'].values, df_tgt['i'].values

        tgt_u_neg = np.array([self.all_u[~self.all_u.isin(df_obs_tgt[df_obs_tgt['i'] == i]['u'])]
                             .sample(self.n_neg_sample).values for i in tgt_i])
        tgt_i_neg = np.array([self.all_i[~self.all_i.isin(df_obs_tgt[df_obs_tgt['u'] == u]['i'])]
                             .sample(self.n_neg_sample).values for u in tgt_u])
        assert 0 not in tgt_u_neg and 0 not in tgt_i_neg and 0 not in tgt_i

        # get users' interactions sequence for short-term interest
        df_obs = self.df.iloc[:idx_tgt_start]

        # get trend mask
        if self.cal_trend:
            mask_trend = self.mask_trend[idx]
        else:
            mask_trend = None

        if not output_diff:
            assert idx == 0
            df_h = df_obs.iloc[:idx_tgt_start]
            adj_obs = self.build_ui_mat(df_h)
            return t_diff, adj_obs, adj_ins, tgt_u, tgt_i, tgt_u_neg, tgt_i_neg, mask_trend
        else:
            assert idx > 0
            idx_obs_diff_start = self.cum_n_records[idx - 1]
            df_h_diff = df_obs.iloc[idx_obs_diff_start:idx_tgt_start]
            adj_obs_diff = self.build_ui_mat(df_h_diff)
            return t_diff, adj_obs_diff, adj_ins, tgt_u, tgt_i, tgt_u_neg, tgt_i_neg, mask_trend

    def build_ui_mat(self, df):
        return sp.csc_matrix((np.ones(len(df)), (df.iloc[:, 0], df.iloc[:, 1])), shape=[self.n_user, self.n_item])

    def get_last_df(self):
        """ get last DataFrame for inherent """
        return self.df


class Dataloader:
    def __init__(self, args, ds):
        self.ds = ds
        self.device = args.device
        self.alpha_spectrum = args.alpha_spectrum
        self.n_batch_load = args.n_batch_load
        self.length = len(self.ds)
        self.n_trend, self.n_non_trend = ds.n_trend, ds.n_non_trend

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.get_iter(0)

    def get_iter(self, start_idx=0):
        return BackgroundGenerator(self._get_iter(start_idx), self.n_batch_load)

    def _get_iter(self, start_idx=0):
        adj_obs, adj_obs_diff = None, None

        for i in range(start_idx, len(self.ds)):
            if adj_obs is None:
                t_diff, adj_obs, adj_ins, tgt_u, tgt_i, tgt_u_neg, tgt_i_neg, mask_trend = \
                    self.ds.getitem(i, output_diff=False)
            else:
                t_diff, adj_obs_diff, adj_ins, tgt_u, tgt_i, tgt_u_neg, tgt_i_neg, mask_trend = \
                    self.ds.getitem(i, output_diff=True)
                adj_obs += adj_obs_diff

            adj_obs_laplacian = biadj_to_laplacian(adj_obs) * self.alpha_spectrum
            adj_ins_i2u, adj_ins_u2i = biadj_to_propagation(adj_ins)
            adj_obs_laplacian, adj_ins_i2u, adj_ins_u2i = \
                [sparse_mx_to_torch_sparse_tensor(v).to(self.device) for v in [adj_obs_laplacian, adj_ins_i2u,
                                                                               adj_ins_u2i]]

            t_diff = torch.FloatTensor([t_diff]).to(self.device)
            tgt_u = torch.from_numpy(tgt_u).long().unsqueeze(-1).to(self.device)
            tgt_i = torch.from_numpy(tgt_i).long().unsqueeze(-1).to(self.device)
            tgt_u_neg = torch.from_numpy(tgt_u_neg).long().to(self.device)
            tgt_i_neg = torch.from_numpy(tgt_i_neg).long().to(self.device)

            if mask_trend is not None:
                mask_trend = mask_trend.long().to(self.device)

            yield t_diff, adj_obs_laplacian, adj_ins_i2u, adj_ins_u2i, tgt_u, tgt_i, tgt_u_neg, tgt_i_neg, mask_trend


def split_data(proportion_train, df):
    """ Split whole data_raw into train, validation and test sets """
    num_interactions = len(df)
    idx_val_start = int(num_interactions * proportion_train)
    idx_test_start = int(num_interactions * (proportion_train + 0.1))
    idx_test_end = int(num_interactions * (proportion_train + 0.2))

    df_tr = df.iloc[:idx_val_start]
    df_val = df.iloc[idx_val_start:idx_test_start]
    df_te = df.iloc[idx_test_start:idx_test_end]

    return df_tr, df_val, df_te


def read_data(args):
    """ check the existence of processed csv, matrix dictionary or raw file """
    # check data
    if not os.path.exists(args.path_csv):
        if os.path.exists(args.path_raw):
            raise FileNotFoundError(f'Raw dataset {args.dataset} found without preprocessed, '
                                    f'please run preprocessed.py first.')
        else:
            raise FileNotFoundError(f'Dataset {args.dataset} not found.')

    # load data, add padding
    df = pd.read_csv(args.path_csv)
    df.columns = ['u', 'i1', 't']
    df['i'] = df['i1'] + 1
    df = df.drop(columns=['i1'])[['u', 'i', 't']]
    args.ts_max = df['t'].max()

    # check dataframe, considering item padding
    assert df['u'].max() + 1 == df['u'].nunique()
    assert df['i'].max() == df['i'].nunique()
    assert (df['t'].diff().iloc[1:] >= 0).all()
    args.n_user, args.n_item = df.iloc[:, :2].max() + 1

    return df


def get_dataloader(args, noter):
    df = read_data(args)
    df_tr, df_val, df_te = split_data(args.proportion_train, df)

    if args.cal_trend:
        if os.path.exists(args.f_mask_trend):
            # load preprocessed mask_trend
            with open(args.f_mask_trend, 'rb') as f:
                mask_trend = pickle.load(f)
            print(f'\n[info] Loading preprocessed mask_trend file.')
        else:
            mask_trend = {'tr': None, 'val': None, 'te': None}
            print(f'\n[info] Failed loading preprocessed mask_trend file, re-creating it...')

    # pack to Dataset
    ds_tr = ISRDataset(args, df_tr, mask_trend=mask_trend['tr'])
    ds_val = ISRDataset(args, df_val, df_0=ds_tr.get_last_df(), mask_trend=mask_trend['val'])
    ds_te = ISRDataset(args, df_te, df_0=ds_val.get_last_df(), mask_trend=mask_trend['te'])
    noter.log_msg(f'\n[info] Dataset')
    noter.log_msg(f'\t| users {args.n_user} | items {args.n_item - 1} | interactions {len(df)} '
                  f'| timestamps {df["t"].nunique()} |'
                  f'\n\t| interactions | train {len(df_tr)} | valid {len(df_val)} | test {len(df_te)} |'
                  f'\n\t| timestamps   | train {len(ds_tr)} | valid {len(ds_val)} | test {len(ds_te)} |')

    # save mask_trend if none
    if not os.path.exists(args.f_mask_trend):
        with open(args.f_mask_trend, 'wb') as f:
            pickle.dump({'tr': ds_tr.mask_trend,
                         'val': ds_val.mask_trend,
                         'te': ds_te.mask_trend}, f)

    # pack to DataLoader
    trainloader = Dataloader(args, ds_tr)
    valloader = Dataloader(args, ds_val)
    testloader = Dataloader(args, ds_te)

    return trainloader, valloader, testloader
