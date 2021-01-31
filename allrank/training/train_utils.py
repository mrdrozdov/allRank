import os
from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

import allrank.models.metrics as metrics_module
from allrank.data.dataset_loading import PADDED_Y_VALUE
from allrank.models.model_utils import get_num_params, log_num_params
from allrank.training.early_stop import EarlyStop
from allrank.utils.ltr_logging import get_logger

logger = get_logger()


def loss_batch(model, loss_func, xb, yb, indices, gradient_clipping_norm, opt=None):
    mask = (yb == PADDED_Y_VALUE)
    loss = loss_func(model(xb, mask, indices), yb, indices=indices)

    if opt is not None:
        loss.backward()
        if gradient_clipping_norm:
            clip_grad_norm_(model.parameters(), gradient_clipping_norm)
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def metric_on_batch(metric, model, xb, yb, indices):
    mask = (yb == PADDED_Y_VALUE)
    return metric(model.score(xb, mask, indices), yb)


def metric_on_epoch(metric, model, feature_func, dl, dev):
    lst = []
    for xb, yb, indices in wrap_dl(dl, feature_func):
        val = metric_on_batch(metric, model, xb.to(device=dev), yb.to(device=dev), indices.to(device=dev))
        lst.append(val)
    metric_values = torch.mean(
        torch.cat(lst), dim=0).cpu().numpy()
    return metric_values


def compute_metrics(metrics, model, feature_func, dl, dev):
    metric_values_dict = {}
    for metric_name, ats in metrics.items():
        metric_func = getattr(metrics_module, metric_name)
        metric_func_with_ats = partial(metric_func, ats=ats)
        metrics_values = metric_on_epoch(metric_func_with_ats, model, feature_func, dl, dev)
        metrics_names = ["{metric_name}_{at}".format(metric_name=metric_name, at=at) for at in ats]
        metric_values_dict.update(dict(zip(metrics_names, metrics_values)))

    return metric_values_dict


def epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics):
    summary = "Epoch : {epoch} Train loss: {train_loss} Val loss: {val_loss}".format(
        epoch=epoch, train_loss=train_loss, val_loss=val_loss)
    for metric_name, metric_value in train_metrics.items():
        summary += " Train {metric_name} {metric_value}".format(
            metric_name=metric_name, metric_value=metric_value)

    for metric_name, metric_value in val_metrics.items():
        summary += " Val {metric_name} {metric_value}".format(
            metric_name=metric_name, metric_value=metric_value)

    return summary


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def wrap_dl(dl, feature_func, return_all=False):
    def res_func(sample):
        if return_all:
            return sample
        else:
            return sample[:3]

    def first_wrap(dl):
        cache = []
        for sample in dl:
            cache.append(sample)
            if len(cache) == feature_func.main_loop_batch:
                yield cache
                cache = []
        if len(cache) > 0:
            yield cache

    def multi_batch(cache):
        if not feature_func.load_in_main_loop:
            for sample in cache:
                yield res_func(sample)

        else:
            mxb, _, _, mqb, _ = zip(*cache)
            new_xb = feature_func._load_in_main_loop(torch.cat(mxb, 0), torch.cat(mqb, 0))
            new_mxb = torch.split(new_xb, mxb[0].shape[0], dim=0)
            for xb, old_xb, sample in zip(new_mxb, mxb, cache):
                _, yb, indices, qb, hb = sample
                hb = old_xb
                sample = xb, yb, indices, qb, hb
                yield res_func(sample)

    for cache in first_wrap(dl):
        for out in multi_batch(cache):
            yield out


def fit(epochs, model, feature_func, loss_func, optimizer, scheduler, train_dl, valid_dl, config,
        gradient_clipping_norm, early_stopping_patience, device, output_dir, tensorboard_output_path):

    num_params = get_num_params(model)
    log_num_params(num_params)

    early_stop = EarlyStop(early_stopping_patience)

    for epoch in range(epochs):
        logger.info("Current learning rate: {}".format(get_current_lr(optimizer)))

        model.train()
        # xb dim: [batch_size, slate_length, embedding_dim]
        # yb dim: [batch_size, slate_length]

        train_losses, train_nums = [], []
        for xb, yb, indices in tqdm(wrap_dl(train_dl, feature_func), desc='tr'):
            loss, num = loss_batch(model, loss_func, xb.to(device=device), yb.to(device=device), indices.to(device=device),
                    gradient_clipping_norm, optimizer)
            train_losses.append(loss)
            train_nums.append(num)

        #train_losses, train_nums = zip(
        #    *[loss_batch(model, loss_func, xb.to(device=device), yb.to(device=device), indices.to(device=device),
        #                 gradient_clipping_norm, optimizer) for
        #      xb, yb, indices in train_dl])
        logger.info('compute loss')
        train_loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
        logger.info('Warning: Skipping train metrics.')
        #train_metrics = compute_metrics(config.metrics, model, feature_func, train_dl, device)
        train_metrics = {}

        model.eval()
        with torch.no_grad():
            val_losses, val_nums = [], []
            for xb, yb, indices in tqdm(wrap_dl(valid_dl, feature_func), desc='va'):
                loss, num = loss_batch(model, loss_func, xb.to(device=device), yb.to(device=device), indices.to(device=device),
                        gradient_clipping_norm)
                val_losses.append(loss)
                val_nums.append(num)
            #val_losses, val_nums = zip(
            #    *[loss_batch(model, loss_func, xb.to(device=device), yb.to(device=device), indices.to(device=device),
            #                 gradient_clipping_norm) for
            #      xb, yb, indices in valid_dl])
            logger.info('Warning: Skipp val metrics.')
            #val_metrics = compute_metrics(config.metrics, model, feature_func, valid_dl, device)
            val_metrics = {}

        val_loss = np.sum(np.multiply(val_losses, val_nums)) / np.sum(val_nums)

        if config.log['save_every_epoch']:
            torch.save(model.state_dict(), os.path.join(output_dir, "model-epoch{}.pkl".format(epoch)))

        logger.info(epoch_summary(epoch, train_loss, val_loss, train_metrics, val_metrics))

        logger.info('Warning: The val metric is hard-coded.')
        #current_val_metric_value = val_metrics.get(config.val_metric)
        current_val_metric_value = 1
        if scheduler:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                args = [val_metrics[config.val_metric]]
                scheduler.step(*args)
            else:
                scheduler.step()

        early_stop.step(current_val_metric_value, epoch)
        if early_stop.stop_training(epoch):
            logger.info(
                "early stopping at epoch {} since {} didn't improve from epoch no {}. Best value {}, current value {}".format(
                    epoch, config.val_metric, early_stop.best_epoch, early_stop.best_value, current_val_metric_value
                ))
            break

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pkl"))

    return {
        "epochs": epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "num_params": num_params
    }
