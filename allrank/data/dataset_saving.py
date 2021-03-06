from typing import Iterable

import numpy as np
from sklearn.datasets import dump_svmlight_file

from allrank.data.dataset_loading import PADDED_Y_VALUE


def write_out(path, out, dstore):
    with open(path, 'w') as f:
        cur_qid = None
        for rank_lst, label_lst, qid_lst, kid_lst, q_src_lst, x_tgt_lst in zip(
            out['rank'], out['label'], out['qid'], out['kid'], out['q_src'], out['x_tgt']):

            rank_lst = rank_lst.view(-1).long().tolist()
            label_lst = label_lst.view(-1).long().tolist()
            qid_lst = qid_lst.view(-1).long().tolist()
            kid_lst = kid_lst.view(-1).long().tolist()
            q_src_lst = q_src_lst.contiguous().view(-1).long().tolist()
            x_tgt_lst = x_tgt_lst.contiguous().view(-1).long().tolist()

            for r, l, q, k, q_src, x_tgt in zip(rank_lst, label_lst, qid_lst, kid_lst, q_src_lst, x_tgt_lst):
                if cur_qid is not None and cur_qid != q:
                    f.write('\n')
                line = ''
                line += '{}'.format(l)
                line += ' {:<16}'.format('qid={}'.format(q))
                #line += ' {:<16}'.format('xid={}'.format(k))
                line += ' {:<16}'.format('rnk={}'.format(r))
                line += ' {:<16}'.format('qtok={}'.format(q_src))
                line += ' {:<16}'.format('xtok={}'.format(x_tgt))
                line += ' {:<16}'.format('qstr={}'.format(dstore.vocab.symbols[q_src]))
                line += ' {:<16}'.format('xstr={}'.format(dstore.vocab.symbols[x_tgt]))
                line += '\n'
                f.write(line)
                cur_qid = q


def write_to_libsvm_without_masked(path: str, X: Iterable[np.ndarray], y: Iterable[np.ndarray]) -> None:
    """
    This function writes given X's and y's in svmlight / libsvm file format.
    It supports padded documents - they are removed from the written dataset.
    Slates are identified by a 'qid' column within the file.

    :param path: a path to save libsvm file
    :param X: Iterable of lists of document vectors
    :param y: Iterable of lists of document relevancies
    """
    Xs = []
    ys = []
    qids = []
    qid = 0
    for X, y in zip(X, y):
        mask = y != PADDED_Y_VALUE
        Xs.append(X[mask])  # type:ignore
        ys.append(y[mask])  # type:ignore
        qids.append(np.repeat(qid, len(y[mask])))  # type:ignore
        qid += 1
    Xs = np.vstack(Xs)
    ys = np.concatenate(ys)
    qids = np.concatenate(qids)
    dump_svmlight_file(Xs, ys, path, query_id=qids)
