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
                       q_path=None, q_dstore_size=None, vocab=None):
        self.path = path
        self.dstore_size = dstore_size
        self.q_path = q_path
        self.q_dstore_size = q_dstore_size
        self.vec_size = vec_size
        self.enabled = enabled
        self.load_in_collate = load_in_collate
        self.load_in_main_loop = load_in_main_loop
        self.load_in_call = not self.load_in_collate and not self.load_in_main_loop
        self.load_xb = load_xb
        self.main_loop_batch = main_loop_batch
        self.prefetch = prefetch
        self._initialized = False
        if vocab is not None:
            self.vocab = Dictionary()
            self.vocab.add_from_file(vocab)
            self.vocab.finalize()

        assert not (self.load_in_collate and self.load_in_main_loop), "Choose one."

    def initialize(self):
        self.keys = np.memmap(os.path.join(self.path, 'dstore_keys.npy'), dtype=np.float32, mode='r', shape=(self.dstore_size, self.vec_size))
        self.q_keys = np.memmap(os.path.join(self.q_path, 'dstore_keys.npy'), dtype=np.float32, mode='r', shape=(self.q_dstore_size, self.vec_size))
        self._initialized = True

    def load_fetch(self, path, keys, index):
        index = list(sorted(index))

        m = hashlib.sha256()
        m.update(str.encode('v0.0.2'))
        for x in index:
            m.update(str.encode('{}'.format(x)))
        data_hash = m.hexdigest()

        cache_path = os.path.join(path, '{}.cache.npy'.format(data_hash))

        if not os.path.exists(cache_path):
            print('build cache and save to {}'.format(cache_path))
            bsz = 1000
            nbatches = len(index) // bsz
            if nbatches * bsz < len(index):
                nbatches += 1
            fetched = []
            for i in tqdm(range(nbatches)):
                start = i * bsz
                end = min(start + bsz, len(index))
                local_index = index[start:end]
                fetched.append(keys[local_index])
            fetched = np.concatenate(fetched, axis=0)
            np.save(cache_path, fetched)
        else:
            print('read cache from {}'.format(cache_path))
            fetched = np.load(cache_path)
        return fetched, index

    def run_prefetch(self, dl_lst):
        if not self._initialized:
            self.initialize()
        print('Run prefetch...')
        unique_ids_x = set()
        unique_ids_q = set()
        for dl in dl_lst:
            for xb, yb, indices, qb, hb in dl:
                unique_ids_q.update(np.unique(qb))
                unique_ids_x.update(np.unique(xb.long()))

        # x
        fetched, index = self.load_fetch(self.path, self.keys, unique_ids_x)
        index = np.asarray(index, dtype=np.int)
        data = np.arange(index.shape[0])
        indices = np.zeros(data.shape[0], dtype=np.int)
        indptr = np.zeros(self.keys.shape[0] + 1, dtype=np.int)
        indptr[index + 1] = 1
        indptr = np.cumsum(indptr)
        sparse_to_dense = scipy.sparse.csr_matrix(
                (data, indices, indptr),
                shape=(self.keys.shape[0], 1), dtype=np.int)
        self.sparse_vecs = fetched
        self.sparse_to_dense = sparse_to_dense

        # q
        fetched, index = self.load_fetch(self.q_path, self.keys, unique_ids_q)
        index = np.asarray(index, dtype=np.int)
        data = np.arange(index.shape[0])
        indices = np.zeros(data.shape[0], dtype=np.int)
        indptr = np.zeros(self.keys.shape[0] + 1, dtype=np.int)
        indptr[index + 1] = 1
        indptr = np.cumsum(indptr)
        sparse_to_dense = scipy.sparse.csr_matrix(
                (data, indices, indptr),
                shape=(self.keys.shape[0], 1), dtype=np.int)
        self.q_sparse_vecs = fetched
        self.q_sparse_to_dense = sparse_to_dense
        print('done.')

    def load_from_memmap(self, idx, feat_type=None):
        u, inv = np.unique(idx.cpu().long().numpy(), return_inverse=True)
        if feat_type == 'x':
            new_idx = self.sparse_to_dense[u].toarray().reshape(-1)
            tmp = self.sparse_vecs[new_idx]
        elif feat_type == 'q':
            new_idx = self.q_sparse_to_dense[u].toarray().reshape(-1)
            tmp = self.q_sparse_vecs[new_idx]
        else:
            raise ValueError

        tmp = tmp[inv]
        tmp = torch.from_numpy(tmp).view(idx.shape[0], idx.shape[1], self.vec_size)
        return tmp

    def load(self, xb, qb):
        xb = self.load_from_memmap(xb, feat_type='x')
        qb = self.load_from_memmap(qb, feat_type='q')
        out = torch.cat([xb, qb], -1)
        return out

    def _load_in_main_loop(self, xb, qb):
        if not self.enabled or not self.load_in_main_loop:
            return xb
        if not self._initialized:
            self.initialize()
        return self.load(xb, qb)

    def _load_in_collate(self, xb, qb):
        if not self.enabled or not self.load_in_collate:
            return xb
        if not self._initialized:
            self.initialize()
        return self.load(xb, qb)

    def __call__(self, xb, qb):
        if not self.enabled or not self.load_in_call:
            return xb
        if not self._initialized:
            self.initialize()
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
    # reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    args = parse_args()

    paths = PathsContainer.from_args(args.job_dir, args.run_id, args.config_file_name)

    create_output_dirs(paths.output_dir)

    logger = init_logger(paths.output_dir)
    logger.info(f"created paths container {paths}")

    # read config
    config = Config.from_json(paths.config_path)
    logger.info("Config:\n {}".format(pformat(vars(config), width=1)))

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
    if dstore.enabled:
        n_features += dstore.vec_size
        n_features += dstore.vec_size - 1 # load xb

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
    model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
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
