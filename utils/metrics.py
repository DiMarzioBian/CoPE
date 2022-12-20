import numpy as np


def cal_mrr(rank_u):
    rank_u = np.array(rank_u)
    return (1 / rank_u).mean()


def cal_recall(rank_u, k):
    rank_u = np.array(rank_u)
    return (rank_u <= k).mean()
