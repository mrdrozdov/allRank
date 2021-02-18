from itertools import product

import torch
from torch.nn import BCEWithLogitsLoss

from allrank.data.dataset_loading import PADDED_Y_VALUE


def get_knn_log_prob(scores, tgts, knn_tgts):
    probs = torch.log_softmax(scores, dim=-1)
    index_mask = torch.eq(knn_tgts, tgts).float()
    index_mask[index_mask == 0] = -10000 # for stability
    index_mask[index_mask == 1] = 0
    yhat_knn_prob = torch.logsumexp(probs + index_mask, dim=-1)
    return yhat_knn_prob.view(scores.shape[0], 1)


def combine_knn_and_vocab_probs(knn_p, vocab_p, coeff):
    combine_probs = torch.stack([vocab_p, knn_p], dim=0)
    coeffs = torch.ones_like(combine_probs)
    coeffs[0] = torch.log(1 - coeff)
    coeffs[1] = torch.log(coeff)
    curr_prob = torch.logsumexp(combine_probs + coeffs, dim=0)
    return curr_prob


def e2e(y_pred, y_true, model_input=None, model=None, padded_value_indicator=PADDED_Y_VALUE, weight_by_diff=False, weight_by_diff_powed=False, indices=None, mode='mean_rank', margin=0.01):
    x_id, q_src, p, dist, q_tgt, x_tgt = torch.chunk(model_input[:, :, -6:], 6, dim=2)
    feat = model_input[:, :, :-6]
    x_feat, q_feat = torch.chunk(feat, 2, dim=-1)
    y_pred = y_pred.clone() # fyi, clone does retain computational graph
    y_true = y_true.clone()
    B, K = y_true.shape

    dist = dist.squeeze(-1)
    p = p[:, 0]
    knn_tgts = x_tgt.squeeze(-1)
    tgts = q_tgt[:, 0]

    assert knn_tgts.shape == (y_true.shape[0], y_true.shape[1])
    assert tgts.shape == (y_true.shape[0], 1)

    mode = 'scores'

    if mode == 'dist':
        knn_scores = -dist # WARNING: This is non-differentiable!
    elif mode == 'scores':
        knn_scores = y_pred
    knn_p = get_knn_log_prob(knn_scores, tgts.long(), knn_tgts.long())

    assert p.shape == (y_true.shape[0], 1)
    assert knn_p.shape == (y_true.shape[0], 1)

    coeff = torch.tensor([0.5], dtype=torch.float, device=knn_p.device).view(1)
    #coeff = torch.sigmoid(model.coeff_layer(q_feat[:, 0]))

    new_p = combine_knn_and_vocab_probs(knn_p, p, coeff)

    loss = -new_p # negative log-likelihood

    return loss.mean()
