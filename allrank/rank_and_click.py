import collections
import os
from argparse import ArgumentParser, Namespace
from pprint import pformat
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from attr import asdict

from allrank.main import Dstore

from allrank.click_models.click_utils import click_on_slates
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset_role, make_dataloader, create_data_loaders, load_libsvm_dataset
from allrank.data.dataset_saving import write_to_libsvm_without_masked, write_out
from allrank.inference.inference_utils import rank_slates, metrics_on_clicked_slates
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel, load_state_dict_from_file
from allrank.utils.args_utils import split_as_strings
from allrank.utils.command_executor import execute_command
from allrank.utils.config_utils import instantiate_from_recursive_name_args
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import all_equal


def parse_args():
    parser = ArgumentParser("allRank rank and apply click model")
    parser.add_argument("--job-dir", help="Base output path for all experiments", required=True)
    parser.add_argument("--run-id", help="Name of this run to be recorded (must be unique within output dir)",
                        required=True)
    parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with model config")
    parser.add_argument("--input-model-path", required=True, type=str, help="Path to the model to read weights")
    parser.add_argument("--roles", required=True, type=split_as_strings,
                        help="List of comma-separated dataset roles to load and process")

    return parser.parse_args()


def run():
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    args = parse_args()

    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

    os.makedirs(paths.base_output_path, exist_ok=True)

    create_output_dirs(paths.output_dir)
    logger = init_logger(paths.output_dir)

    logger.info("will save data in {output_dir}".format(output_dir=paths.base_output_path))

    # read config
    config = Config.from_json(paths.config_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    output_config_path = os.path.join(paths.output_dir, "used_config.json")
    execute_command("cp {} {}".format(paths.config_path, output_config_path))

    train_ds, val_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    )

    # load dstore and use as feature func
    dstore = Dstore(**config.dstore)
    n_features = train_ds.shape[-1]
    n_features = dstore.get_n_features(n_features, config)

    train_dl, val_dl = create_data_loaders(
        train_ds, val_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size,
        dstore=dstore)

    if dstore.prefetch:
        dstore.run_prefetch([train_dl, val_dl])

    # gpu support
    dev = get_torch_device()
    logger.info("Will use device {}".format(dev.type))

    # instantiate model
    model = make_model(n_features=n_features, dstore=dstore, **asdict(config.model, recurse=False))

    model.load_state_dict(load_state_dict_from_file(args.input_model_path, dev))
    logger.info(f"loaded model weights from {args.input_model_path}")

    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
    model.to(dev)

    datasets = {'vali': val_dl}

    ranked_slates = rank_slates(datasets, model, dstore, config)

    # save output
    for role, out in ranked_slates.items():
        write_out_dir(paths.output_dir, role, out, dstore)

    print('DONE')


def write_out_dir(path, role, out, dstore):
    print('write to {}'.format(path))
    keys = ['knn_rank', 'query_id', 'knns', 'knn_tgts', 'scores']
    out['knns'] = out['kid']
    out['knn_tgts'] = out['x_tgt']
    out['knn_rank'] = out['rank']
    out['query_id'] = out['qid']
    #
    new_out = collections.OrderedDict()
    for k in keys:
        assert len(out[k]) > 0, k
        new_out[k] = torch.cat(out[k], 0).unsqueeze(-1).numpy()
    new_out['query_id'] = new_out['query_id'][:, 0]
    # revert cache ids
    shape = new_out['query_id'].shape
    new_out['query_id'] = dstore.unique_q[new_out['query_id'].reshape(-1)].reshape(*shape)
    shape = new_out['knns'].shape
    new_out['knns'] = dstore.unique_x[new_out['knns'].reshape(-1)].reshape(*shape)
    #
    for k, v in new_out.items():
        dtype = np.float32 if k == 'scores' else np.int
        new_path = os.path.join(path, 'out_{}_{}.npy'.format(role, k))
        print('writing {} with shape {}'.format(new_path, v.shape))
        x = np.memmap(new_path, mode='w+', dtype=dtype, shape=v.shape)
        x[:] = v


if __name__ == "__main__":
    run()
