import torch
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error


def calc_recall(rank, ground_truth, k):
    """
    calculate recall of one example
    """
    return len(set(rank[:k]) & set(ground_truth)) / float(len(set(ground_truth)))

def hit_at_k(hit, k):
    """
    calculate Hit@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)[:k]
    return int(np.any(hit[:]==1))


def hit_at_k_batch(hits, k):
    """
    calculate Hit@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    hits = hits[:, :k]
    res = []
    for i in range(len(hits)):
        res.append(int(np.any(hits[i][:]==1)))
    return np.array(res)


def precision_at_k(hit, k):
    """
    calculate Precision@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)[:k]
    return np.mean(hit)


def precision_at_k_batch(hits, k):
    """
    calculate Precision@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = hits[:, :k].mean(axis=1)
    return res


def average_precision(hit, cut):
    """
    calculate average precision (area under PR curve)
    hit: list, element is binary (0 / 1)
    """
    hit = np.asarray(hit)
    precisions = [precision_at_k(hit, k + 1) for k in range(cut) if len(hit) >= k]
    if not precisions:
        return 0.
    return np.sum(precisions) / float(min(cut, np.sum(hit)))


def dcg_at_k(rel, k):
    """
    calculate discounted cumulative gain (dcg)
    rel: list, element is positive real values, can be binary
    """
    rel = np.asfarray(rel)[:k]
    dcg = np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2)))
    return dcg


def ndcg_at_k(rel, k):
    """
    calculate normalized discounted cumulative gain (ndcg)
    rel: list, element is positive real values, can be binary
    """
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(rel, k) / idcg


def ndcg_at_k_batch(hits, k):
    """
    calculate NDCG@k
    hits: array, element is binary (0 / 1), 2-dim

    return: 1D array
    """
    hits_k = hits[:, :k]
    dcg = np.sum((2 ** hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)
    idcg[idcg == 0] = np.inf

    res = (dcg / idcg)
    return res


def recall_at_k(hit, k, all_pos_num):
    """
    calculate Recall@k
    hit: list, element is binary (0 / 1)
    """
    hit = np.asfarray(hit)[:k]
    return np.sum(hit) / all_pos_num


def recall_at_k_batch(hits, k):
    """
    calculate Recall@k
    hits: array, element is binary (0 / 1), 2-dim
    """
    res = (hits[:, :k].sum(axis=1) / hits.sum(axis=1))
    return res


def collision_ratio(list1, list2, k):
    return sum(x == y for x, y in zip(list1[:k], list2[:k])) / k


def collision_ratio_batch(list1, list2, k):
    assert len(list1) == len(list2)
    rst = 0
    for i in range(len(list1)):
        rst += sum(x == y for x, y in zip(list1[i][:k], list2[i][:k])) / k
    return rst / len(list1)


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def calc_auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res


def logloss(ground_truth, prediction):
    logloss = log_loss(np.asarray(ground_truth), np.asarray(prediction))
    return logloss


def calc_metrics_med(cf_scores, train_pt_tx_dict, test_pt_tx_dict, pt_ids_batch, tx_ids_numpy):
    """
    cf_scores: (n_batch_patients, n_all_treatments)
    """
    tx_id_map = {}  # {org ent-id: 0-based id}
    for i, tx_id in enumerate(tx_ids_numpy):
        tx_id_map[tx_id] = i
    test_pos_tx_binary = np.zeros([len(pt_ids_batch), len(tx_ids_numpy)], dtype=np.float32)
    for idx, pt_id in enumerate(pt_ids_batch):
        # train_pos_tx_list = train_pt_tx_dict[pt_id]
        # cf_scores[idx][train_pos_tx_list] = 0
        for tx_id in test_pt_tx_dict[pt_id]:
            test_pos_tx_binary[idx][tx_id_map[tx_id]] = 1

    try:
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)    # try to speed up the sorting process
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(pt_ids_batch)):
        binary_hit.append(test_pos_tx_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)

    return binary_hit


def calc_metrics_com(cf_scores, train_pt_tx_dict, test_pt_tx_dict, test_user_ids_batch, item_ids_numpy):
    """
    cf_scores: (n_batch_users, n_all_items)
    test_user_ids_batch: (n_batch_test)
    """
    item_id_map = {}  # {org ent-id: 0-based id}
    for i, i_id in enumerate(item_ids_numpy):
        item_id_map[i_id] = i
    test_pos_item_binary = np.zeros([len(test_user_ids_batch), len(item_ids_numpy)], dtype=np.float32)
    for idx, u_id in enumerate(test_user_ids_batch):
        train_pos_tx_list = train_pt_tx_dict[u_id]
        cf_scores[idx][train_pos_tx_list] = 0
        for i_id in test_pt_tx_dict[u_id]:
            test_pos_item_binary[idx][item_id_map[i_id]] = 1

    try:
        _, rank_indices = torch.sort(cf_scores.cuda(), descending=True)    # try to speed up the sorting process
    except:
        _, rank_indices = torch.sort(cf_scores, descending=True)
    rank_indices = rank_indices.cpu()

    binary_hit = []
    for i in range(len(test_user_ids_batch)):
        binary_hit.append(test_pos_item_binary[i][rank_indices[i]])
    binary_hit = np.array(binary_hit, dtype=np.float32)

    return binary_hit

def get_Ks(K: int):
    assert K > 0
    Ks = []
    for n in [1, 3, 5, 10, 20]:
        if K >= n :
            Ks.append(n)
    return Ks
