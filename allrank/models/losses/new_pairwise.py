import torch

from allrank.data.dataset_loading import PADDED_Y_VALUE


def hinge_loss(y_i, y_i_rank, y_j, y_j_rank, margin=1):
    loss = torch.relu(margin - torch.sign(y_i_rank - y_j_rank) * (y_j - y_i))
    return loss


def find_max_length(y_true, pad):
    return (y_true.shape[-1] - (y_true == pad).sum(-1)).max().item()


def new_pairwise(y_pred, y_true, indices=None, num_samples=-1, padded_value_indicator=PADDED_Y_VALUE):
    """
    Pointwise RMSE loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param no_of_levels: number of unique ground truth values
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    batch_size = y_true.shape[0]
    y_pred = y_pred.clone()
    y_true = y_true.clone()
    device = y_true.device

    loss = []
    for i in range(batch_size):
        _y_true = y_true[i]
        _y_pred = y_pred[i]
        max_length = find_max_length(_y_true, padded_value_indicator)
        index = torch.combinations(torch.arange(max_length))
        #y_rank = torch.arange(max_length).to(device).float()
        y_rank = indices[i].float()

        y_i = _y_pred[index[:, 0]]
        y_i_rank = y_rank[index[:, 0]]
        y_j = _y_pred[index[:, 1]]
        y_j_rank = y_rank[index[:, 1]]
        loss.append(hinge_loss(y_i, y_i_rank, y_j, y_j_rank, margin=1))

    loss = torch.cat(loss).mean()

    return loss

