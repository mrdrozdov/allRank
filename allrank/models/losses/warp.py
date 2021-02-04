from itertools import product

import torch
from torch.nn import BCEWithLogitsLoss

from allrank.data.dataset_loading import PADDED_Y_VALUE


def rankNet_weightByGTDiff(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE):
    """
    Wrapper for RankNet employing weighing by the differences of ground truth values.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    return rankNet(y_pred, y_true, padded_value_indicator, weight_by_diff=True)


def rankNet_weightByGTDiff_pow(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE):
    """
    Wrapper for RankNet employing weighing by the squared differences of ground truth values.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    return rankNet(y_pred, y_true, padded_value_indicator, weight_by_diff=False, weight_by_diff_powed=True)


def warp(y_pred, y_true, padded_value_indicator=PADDED_Y_VALUE, weight_by_diff=False, weight_by_diff_powed=False, indices=None, mode='mean_rank'):
    """
    TODO: Add margin.
    """
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
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]
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
    max_N = the_mask.shape[-1] // 2

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

        # This isn't necessary, but will terminate early if done.
        if done_vec.all().item():
            break

    def l_func(x):
        # TODO: Handle mode here.
        return x.detach()

    # TODO: What if mask is empty?
    m = (Y_vec > 0) & (done_vec == True) # mask for valid examples.
    energy = -pred_diffs.gather(index=pair_vec.view(-1, 1), dim=1)

    try:
        assert (energy[m] >= 0).all().item(), energy[m]
    except:
        import ipdb; ipdb.set_trace()
        pass
    loss = l_func(Y_vec[m] // N_vec[m]).view(-1, 1) * energy[m]
    assert loss.shape[1] == 1

    return loss.sum()