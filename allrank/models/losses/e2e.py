from itertools import product

import torch
from torch.nn import BCEWithLogitsLoss

from allrank.data.dataset_loading import PADDED_Y_VALUE


def e2e(y_pred, y_true, model_input=None, padded_value_indicator=PADDED_Y_VALUE, weight_by_diff=False, weight_by_diff_powed=False, indices=None, mode='mean_rank', margin=0.01):
    """
    TODO: Add margin.
    """
    x_id, q_src, p, dist, q_tgt, x_tgt = torch.chunk(model_input[:, :, -6:], 6, dim=2)
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    B, K = y_true.shape

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1] - margin
    is_correct = true_diffs.sign() == pred_diffs.sign() # TODO: This will change with margin.

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    # Number of viable pairs per batch. Sometimes this is zero!
    Y_vec = the_mask.sum(dim=1)

    probs = the_mask / Y_vec.view(-1, 1)
    probs[Y_vec == 0] = 1 / the_mask.shape[-1] # handles nan from 0/0
    dist = torch.distributions.categorical.Categorical(probs=probs)

    N_vec = Y_vec.clone().fill_(0)
    done_vec = N_vec.clone().bool().fill_(False)
    done_vec[Y_vec == 0] = True
    pair_vec = N_vec.clone().long().fill_(0)
    max_N = Y_vec.max().item()

    # TODO: What if all done already?

    for step in range(max_N):
        pair_indices = dist.sample()
        pair_correct = is_correct.gather(index=pair_indices.view(-1, 1), dim=1)

        true_pair_diffs = true_diffs.gather(index=pair_indices.view(-1, 1), dim=1)
        true_pair_diffs_ = true_pair_diffs[done_vec == False]
        assert (true_pair_diffs_ > 0).all().item()

        # Add indices.
        pair_mask = done_vec & (pair_correct.view(-1) == False)
        pair_vec[pair_mask] = pair_indices[pair_mask]

        # Update N if not done.
        N_vec[done_vec == False] += 1
        # Update done if found incorrect pair.
        done_vec[pair_correct.view(-1) == False] = True
        # Update done if hit max_N.
        done_vec[step >= Y_vec - 1] = True

        # This isn't necessary, but will terminate early if done.
        if done_vec.all().item():
            break

    def l_func(Y_vec, N_vec):
        assert (Y_vec >= N_vec).all().item(), torch.cat([Y_vec.view(-1, 1), N_vec.view(-1, 1)], 1)
        rank = Y_vec // N_vec # we don't subtract 1 from Y because we work with pairs
        assert (rank > 0).all().item(), rank
        if mode == 'mean_rank':
            out = rank / Y_vec
        elif mode == 'proportion':
            out = N_vec.clone().fill_(1)
        elif mode == 'precision':
            alpha = torch.flip(torch.arange(0, 25, 8) / torch.arange(0, 25, 8).sum(), [-1])
            pad = rank.max().item() - alpha.shape[-1]
            if pad > 0:
                alpha = torch.cat([alpha, torch.zeros(pad)], -1)
            alpha = alpha.to(N_vec.device).cumsum(-1)
            out = alpha[rank - 1]
        return out.detach()

    # TODO: What if mask is empty?
    m = (Y_vec > 0) & (done_vec == True) # mask for valid examples.
    energy = -pred_diffs.gather(index=pair_vec.view(-1, 1), dim=1)
    loss = l_func(Y_vec[m], N_vec[m]).view(-1, 1) * energy[m]
    assert loss.shape[1] == 1

    return loss.sum()
