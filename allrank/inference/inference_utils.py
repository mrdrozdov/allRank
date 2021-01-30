from typing import Tuple, Dict, List, Generator
import collections

import torch
from torch.utils.data.dataloader import DataLoader

from allrank.training.train_utils import wrap_dl

import allrank.models.losses as losses
from allrank.config import Config
from allrank.data.dataset_loading import LibSVMDataset
from allrank.models.metrics import ndcg, dcg
from allrank.models.model import LTRModel
from allrank.models.model_utils import get_torch_device


def rank_slates(datasets, model: LTRModel, dstore, config: Config):
    """
    Ranks given datasets according to a given model

    :param datasets: dictionary of role -> dataset that will be ranked
    :param model: a model to use for scoring documents
    :param config: config for DataLoaders
    :return: dictionary of role -> ranked dataset
        every dataset is a Tuple of torch.Tensor - storing X and y in the descending order of the scores.
    """

    #dataloaders = {role: __create_data_loader(ds, config) for role, ds in datasets.items()}
    dataloaders = datasets

    ranked_slates = {role: __rank_slates(dl, model, dstore) for role, dl in dataloaders.items()}

    return ranked_slates


#def __create_data_loader(ds: LibSVMDataset, config: Config) -> DataLoader:
#    return DataLoader(ds, batch_size=config.data.batch_size, num_workers=config.data.num_workers, shuffle=False)


def __rank_slates(dataloader: DataLoader, model: LTRModel, dstore):
    reranked_X = []
    reranked_y = []
    model.eval()
    device = get_torch_device()

    out = collections.defaultdict(list)

    with torch.no_grad():
        for xb, yb, indices, qb, hb in wrap_dl(dataloader, dstore, return_all=True):
            q_src = xb[:, :, -2].long().to(device)
            x_tgt = xb[:, :, -1].long().to(device)
            rank = indices.to(device=device)
            y_true = yb.to(device=device)

            input_indices = torch.ones_like(y_true).type(torch.long)
            mask = (y_true == losses.PADDED_Y_VALUE)
            scores = model.score(xb.to(device), mask, input_indices)

            scores[mask] = float('-inf')

            _, indices = scores.sort(descending=True, dim=-1)
            res_y = torch.gather(y_true, dim=1, index=indices).cpu()
            res_rank = torch.gather(rank, dim=1, index=indices).cpu()
            res_q_src = torch.gather(q_src, dim=1, index=indices).cpu()
            res_x_tgt = torch.gather(x_tgt, dim=1, index=indices).cpu()

            out['rank'].append(res_rank)
            out['label'].append(res_y)
            out['qid'].append(qb)
            out['kid'].append(qb) # TODO
            out['q_src'].append(res_q_src)
            out['x_tgt'].append(res_x_tgt)

    return out


def __clicked_ndcg(ordered_clicks: List[int]) -> float:
    return ndcg(torch.arange(start=len(ordered_clicks), end=0, step=-1, dtype=torch.float32)[None, :],
                torch.tensor(ordered_clicks)[None, :]).item()


def __clicked_dcg(ordered_clicks: List[int]) -> float:
    return dcg(torch.arange(start=len(ordered_clicks), end=0, step=-1, dtype=torch.float32)[None, :],
               torch.tensor(ordered_clicks)[None, :]).item()


def metrics_on_clicked_slates(clicked_slates: Tuple[List[torch.Tensor], List[List[int]]]) \
        -> Generator[Dict[str, float], None, None]:
    Xs, ys = clicked_slates
    for X, y in zip(Xs, ys):
        yield {
            "slate_length": len(y),
            "no_of_clicks": sum(y > 0),  # type: ignore
            "dcg": __clicked_dcg(y),
            "ndcg": __clicked_ndcg(y)
        }
