from urllib.parse import urlparse
import hashlib
import collections

from tqdm import tqdm

import allrank.models.losses as losses
import numpy as np
import os
import torch
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, CustomDataParallel
from allrank.training.train_utils import fit
from allrank.utils.command_executor import execute_command
from allrank.utils.experiments import dump_experiment_result, assert_expected_metrics
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
from allrank.utils.ltr_logging import init_logger
from allrank.utils.python_utils import dummy_context_mgr
from argparse import ArgumentParser, Namespace
from attr import asdict
from functools import partial
from pprint import pformat
from torch import optim
import scipy


def parse_args() -> Namespace:
    parser = ArgumentParser("allRank")
    parser.add_argument("--job-dir", help="Base output path for all experiments", required=True)
    parser.add_argument("--run-id", help="Name of this run to be recorded (must be unique within output dir)",
                        required=True)
    parser.add_argument("--config-file-name", required=True, type=str, help="Name of json file with config")

    return parser.parse_args()


class Dstore:
    def __init__(self, path, dstore_size=None, vec_size=None, enabled=False, load_in_collate=False, load_in_main_loop=False, main_loop_batch=None, load_xb=False, prefetch=False,
                       q_path=None, q_dstore_size=None, vocab=None, init_from_fasttext=False, fasttext_path=None):
        self.path = path
        self.dstore_size = dstore_size
        self.q_path = q_path
        self.q_dstore_size = q_dstore_size
        self.vec_size = vec_size
        self.enabled = enabled
        self.load_xb = load_xb
        self.load_in_main_loop= load_in_main_loop
        self.main_loop_batch = main_loop_batch
        self.prefetch = prefetch
        self._initialized = False
        if vocab is not None:
            self.vocab = Dictionary()
            self.vocab.add_from_file(vocab)
            self.vocab.finalize()
            print('Found vocab with size {} at path {}'.format(len(self.vocab), vocab))
        self.init_from_fasttext = init_from_fasttext
        self.fasttext_path = fasttext_path
        self.initialize()

    def initialize(self):
        self.keys = np.memmap(os.path.join(self.path, 'dstore_keys.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, self.vec_size))
        self.q_keys = np.memmap(os.path.join(self.q_path, 'dstore_keys.npy'), dtype=np.float32, mode='r', shape=(self.q_dstore_size, self.vec_size))
        self._initialized = True

    def get_n_features(self, n_features, config):
        if self.enabled:
            n_features -= 1 # x_id (key id)
            n_features -= 1 # q_src (query src)
            n_features -= 1 # p
            n_features -= 1 # dist
            n_features -= 1 # q_tgt (query tgt)
            n_features -= 1 # x_tgt (key tgt)
        if not config.model.fc_model['ignore_q_feat']:
            n_features += self.vec_size # query vector
        if not config.model.fc_model['ignore_x_feat']:
            n_features += self.vec_size # key vector
        if config.model.fc_model['embed_q_src']:
            n_features += config.model.fc_model['embed_size']
        if config.model.fc_model['embed_x_tgt']:
            n_features += config.model.fc_model['embed_size']
        return n_features

    def load_fetch(self, path, keys, index):
        m = hashlib.sha256()
        m.update(str.encode('v0.0.4'))
        for x in index:
            m.update(str.encode('{}'.format(x)))
        data_hash = m.hexdigest()

        cache_path = os.path.join(path, '{}.cache.npy'.format(data_hash))

        print('cache source shape {}'.format(keys.shape))

        shape = (len(index), self.vec_size)
        if not os.path.exists(cache_path):
            shape = (len(index), self.vec_size)
            print('build cache shape = {}, and save to {}'.format(shape, cache_path))
            cache = np.memmap(cache_path, mode='w+', dtype=np.float32, shape=shape)
            bsz = 1000
            nbatches = len(index) // bsz
            if nbatches * bsz < len(index):
                nbatches += 1
            for i in tqdm(range(nbatches)):
                start = i * bsz
                end = min(start + bsz, len(index))
                local_index = index[start:end]
                cache[start:end] = keys[local_index]
            del cache
        print('read cache from {}'.format(cache_path))
        cache = np.memmap(cache_path, mode='r', dtype=np.float32, shape=shape)
        return cache, index

    def run_prefetch(self, dl_lst):
        print('Run prefetch...')
        unique_ids_x = set()
        unique_ids_q = set()

        all_lst = collections.defaultdict(list)
        for dl in dl_lst:
            n, k, n_sparse_feat = dl.dataset.shape
            all_lst['x'].append(np.concatenate(dl.dataset.X_by_qid, axis=0).reshape(n, k, n_sparse_feat)[:, :, 0])
            all_lst['q'].append(np.concatenate(dl.dataset.q_by_qid, axis=0).reshape(n, k, 1)[:, :, 0])
            all_lst['dl'].append(dl)
            #knns = all_x[:, :, 0]
            #knn_tgts = all_x[:, :, 3]
            #query_ids = all_q[:, :, 0]

            #unique_ids_q.update(query_ids)
            #unique_ids_x.update(knns.astype(np.int))

        # x
        ids = np.concatenate(all_lst['x'], axis=0).astype(np.int)
        u, inv = np.unique(ids, return_inverse=True)
        fetched, index = self.load_fetch(self.path, self.keys, u)

        self.unique_x = u
        self.x_vecs = fetched
        offset = 0
        for x, dl in zip(all_lst['x'], all_lst['dl']):
            for bucket in dl.dataset.X_by_qid:
                size, _ = bucket.shape
                bucket[:, 0] = inv[offset:offset + size]
                offset += size

        # q
        ids = np.concatenate(all_lst['q'], axis=0).astype(np.int)
        u, inv = np.unique(ids, return_inverse=True)
        fetched, index = self.load_fetch(self.q_path, self.q_keys, u)

        self.unique_q = u
        self.q_vecs = fetched
        offset = 0
        for x, dl in zip(all_lst['x'], all_lst['dl']):
            for bucket in dl.dataset.q_by_qid:
                size = bucket.shape[0]
                bucket[:] = inv[offset:offset + size]
                offset += size

        print('done.')

    def load_from_memmap(self, idx, feat_type=None):
        u, inv = np.unique(idx.cpu().long().numpy(), return_inverse=True)
        if feat_type == 'x':
            tmp = self.x_vecs[u]
        elif feat_type == 'q':
            tmp = self.q_vecs[u]
        else:
            raise ValueError

        tmp = tmp[inv]
        tmp = torch.from_numpy(tmp).view(idx.shape[0], idx.shape[1], self.vec_size)
        return tmp

    def load(self, xb, qb):
        x_id, q_src, p, dist, q_tgt, x_tgt = torch.chunk(xb, 6, dim=2)
        xvec = self.load_from_memmap(x_id.long(), feat_type='x')
        qvec = self.load_from_memmap(qb.long(), feat_type='q')
        out = torch.cat([xvec, qvec, xb], -1)
        return out

    def _load_in_main_loop(self, xb, qb):
        return self.load(xb, qb)

    def _load_in_collate(self, xb, qb):
        return self.load(xb, qb)


class Dictionary(object):
    """
    A mapping from symbols to consecutive integers.

    Taken from fairseq repo.
    """

    def __init__(
        self,
        pad="<pad>",
        eos="</s>",
        unk="<unk>",
        bos="<s>",
        extra_special_symbols=None,
    ):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.bos_index = self.add_symbol(bos)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1, padding_factor=8):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
            - padding_factor can be used to pad the dictionary size to be a
                multiple of 8, which is important on some hardware (e.g., Nvidia
                Tensor Cores).
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[: self.nspecial]
        new_count = self.count[: self.nspecial]

        c = collections.Counter(
            dict(
                sorted(zip(self.symbols[self.nspecial :], self.count[self.nspecial :]))
            )
        )
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

        self.pad_to_multiple_(padding_factor)

    def pad_to_multiple_(self, padding_factor):
        """Pad Dictionary size to be a multiple of *padding_factor*."""
        if padding_factor > 1:
            i = 0
            while len(self) % padding_factor != 0:
                symbol = "madeupword{:04d}".format(i)
                self.add_symbol(symbol, n=0)
                i += 1

    def bos(self):
        """Helper to get index of beginning-of-sentence symbol"""
        return self.bos_index

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    def add_from_file(self, f):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        if isinstance(f, str):
            try:
                with open(f, "r", encoding="utf-8") as fd:
                    self.add_from_file(fd)
            except FileNotFoundError as fnfe:
                raise fnfe
            except UnicodeError:
                raise Exception(
                    "Incorrect encoding detected in {}, please "
                    "rebuild the dataset".format(f)
                )
            return

        for line in f.readlines():
            idx = line.rfind(" ")
            if idx == -1:
                raise ValueError(
                    "Incorrect dictionary format, expected '<token> <cnt>'"
                )
            word = line[:idx]
            count = int(line[idx + 1 :])
            self.indices[word] = len(self.symbols)
            self.symbols.append(word)
            self.count.append(count)


def run():

    args = parse_args()

    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

    create_output_dirs(paths.output_dir)

    logger = init_logger(paths.output_dir)
    logger.info(f"created paths container {paths}")

    # read config
    config = Config.from_json(paths.config_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

    # reproducibility
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    logger.info("Seed: {}".format(config.seed))

    output_config_path = os.path.join(paths.output_dir, "used_config.json")
    execute_command("cp {} {}".format(paths.config_path, output_config_path))

    # train_ds, val_ds
    train_ds, val_ds = load_libsvm_dataset(
        input_path=config.data.path,
        slate_length=config.data.slate_length,
        validation_ds_role=config.data.validation_ds_role,
    )

    n_features = train_ds.shape[-1]
    assert n_features == val_ds.shape[-1], "Last dimensions of train_ds and val_ds do not match!"

    # load dstore and use as feature func
    dstore = Dstore(**config.dstore)
    n_features = dstore.get_n_features(n_features, config)

    # train_dl, val_dl
    train_dl, val_dl = create_data_loaders(
        train_ds, val_ds, num_workers=config.data.num_workers, batch_size=config.data.batch_size,
        dstore=dstore)

    if dstore.prefetch:
        dstore.run_prefetch([train_dl, val_dl])

    # gpu support
    dev = get_torch_device()
    logger.info("Model training will execute on {}".format(dev.type))

    # instantiate model
    model = make_model(n_features=n_features, dstore=dstore, **asdict(config.model, recurse=False))
    if torch.cuda.device_count() > 1:
        model = CustomDataParallel(model)
        logger.info("Model training will be distributed to {} GPUs.".format(torch.cuda.device_count()))
    model.to(dev)

    # load optimizer, loss and LR scheduler
    optimizer = getattr(optim, config.optimizer.name)(params=model.parameters(), **config.optimizer.args)
    loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
    if config.lr_scheduler.name:
        scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(optimizer, **config.lr_scheduler.args)
    else:
        scheduler = None

    with torch.autograd.detect_anomaly() if config.detect_anomaly else dummy_context_mgr():  # type: ignore
        # run training
        result = fit(
            model=model,
            feature_func=dstore,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dl=train_dl,
            valid_dl=val_dl,
            config=config,
            device=dev,
            output_dir=paths.output_dir,
            tensorboard_output_path=paths.tensorboard_output_path,
            **asdict(config.training)
        )

    dump_experiment_result(args, config, paths.output_dir, result)

    if urlparse(args.job_dir).scheme == "gs":
        copy_local_to_gs(paths.local_base_output_path, args.job_dir)

    assert_expected_metrics(result, config.expected_metrics)


if __name__ == "__main__":
    run()
