from os.path import basename, join
import glob
import json

import numpy as np
from scipy import sparse as sp

import h5py

from tqdm import tqdm

from .extractors.base import TopicFeature


def split_docs(X, train_ratio=0.8, keep_format=False, return_idx=True):
    """ Split given documents in train / test

    Inputs:
        X (scipy.sparse.csr_matrix): sparse matrix of document-term relationship
        train_ratio (float): ratio of training samples
        keep_format (bool): whether keeping its original format
        return_idx (bool): whether returning the indices

    Returns:
        scipy.sparse.csr_matrix: train matrix
        scipy.sparse.csr_matrix: test matrix
    """
    if not sp.isspmatrix_csr(X):
        org_fmt = X.format
        X = X.tocsr()

    # split the data
    N = X.shape[0]
    idx = np.random.permutation(N)
    train_idx = idx[:int(train_ratio * N)]
    test_idx = idx[int(train_ratio * N):]

    Xtr = X[train_idx]
    Xts = X[test_idx]

    if keep_format:
        output = (Xtr.asformat(org_fmt), Xts.asformat(org_fmt))
    else:
        output = (Xtr, Xts)

    if return_idx:
        return output + (train_idx, test_idx)
    else:
        return output


def load_csr_data(h5py_fn, row='users', col='items'):
    """ Load recsys data stored in hdf format

    Inputs:
        fn (str): filename for the data

    Returns:
        scipy.sparse.csr_matrix: user-item matrix
        numpy.ndarray: user list
        numpy.ndarray: item list
    """
    with h5py.File(h5py_fn, 'r') as hf:
        data = (hf['data'][:], hf['indices'][:], hf['indptr'][:])
        X = sp.csr_matrix(data)
        rows = hf[row][:]
        cols = hf[col][:]
    return X, rows, cols


def save_feature_h5(features, out_fn):
    """ Save extracted feature to the disk in HDF format

    Inputs:
        features (dict[string] -> TextFeature): extracted features
        out_fn (string): filename to dump the extracted features
    """
    if len(features) == 0:
        raise ValueError('[ERROR] No features found!')

    ids = list(features.values())[0].ids  # anchor
    with h5py.File(out_fn, 'w') as hf:
        hf.create_group('features')
        for key, feat in features.items():
            hf['features'].create_dataset(
                key, data=feat.features[[feat.inv_ids[i] for i in ids]]
            )
            hf['features'].create_dataset(
                key + '_cols',
                data=np.array(feat.columns, dtype=h5py.special_dtype(vlen=str))
            )

            if isinstance(feat, TopicFeature):
                id2token = {token:i for i, token in feat.id2word.items()}
                hf['features'].create_dataset(
                    'topic_terms', data=feat.topic_terms
                )
                hf['features'].create_dataset(
                    'id2word',
                    data=np.array(
                        [id2token[i] for i in range(len(id2token))],
                        dtype=h5py.special_dtype(vlen=str)
                    )
                )
        hf['features'].create_dataset(
            'ids', data=np.array(ids, dtype=h5py.special_dtype(vlen=str))
        )


def save_feature_csv(features, out_fn, delim=','):
    """ Save extracted feature to the disk in csv format

    Inputs:
        features (dict[string] -> TextFeature): extracted features
        out_fn (string): filename to dump the extracted features
        delim (string): delimiter for the separation
    """
    # aggregation process
    ids = list(features.values())[0].ids  # anchor
    # first aggregate the data
    agg_feature = []
    agg_feat_cols = []
    for key, feat in features.items():
        x = feat.features[[feat.inv_ids[i] for i in ids]]
        agg_feature.append(x)
        agg_feat_cols.extend([key + '_' + col for col in feat.columns])
    agg_feature = np.hstack(agg_feature)

    with open(out_fn, 'w') as f:
        f.write(delim.join(agg_feat_cols) + '\n')
        for row in agg_feature:
            f.write(delim.join(['{:.8f}'.format(y) for y in row]) + '\n')


def normalize_matrix(a, order=2, axis=1, eps=1e-12):
    """
    """
    norm = np.linalg.norm(a, order, axis=axis)
    norm = np.maximum(norm, eps)
    if axis == 0:
        return a / norm[None]
    elif axis == 1:
        return a / norm[:, None]
    else:
        raise ValueError('[ERROR] axis should be either 0 or 1!')


def integrate_audio_feat(features, audio_h5, mxm2msd):
    """
    """
    # TODO: this part should be moved to MFCC feature extraction
    #       and stored in the feature file for better integrity
    n_coeffs = 40
    audio_feat_cols = (
        ['mean_mfcc{:d}'.format(i) for i in range(n_coeffs)] +
        ['var_mfcc{:d}'.format(i) for i in range(n_coeffs)] +
        ['mean_dmfcc{:d}'.format(i) for i in range(n_coeffs)] +
        ['var_dmfcc{:d}'.format(i) for i in range(n_coeffs)] +
        ['mean_ddmfcc{:d}'.format(i) for i in range(n_coeffs)] +
        ['var_ddmfcc{:d}'.format(i) for i in range(n_coeffs)]
    )
    with h5py.File(audio_h5, 'r') as hf:
        tid2row = {tid:i for i, tid in enumerate(hf['tids'][:])}
        feats = []
        for mxmid in corpus.ids:
            tid = mxm2msd[mxmid]
            if tid in tid2row:
                feats.append(hf['feature'][tid2row[tid]][None])
            else:
                feats.append(np.zeros((1, len(audio_feat_cols))))
        audio_feat = np.concatenate(feats, axis=0)
        # idx = [tid2row[mxm2msd[mxmid]] for mxmid in corpus.ids]
        # audio_feat = hf['feature'][idx]
        features['audio'] = TextFeature(
            'mfcc', corpus.ids, audio_feat, audio_feat_cols
        )

    return features


def load_mxm_lyrics(fn):
    """ Load a MusixMatch api response

    Read API (track_lyrics_get_get) response.

    Inputs:
        fn (str): filename

    Returns:
        list of string: lines of lyrics
        string: musixmatch tid
    """
    d = json.load(open(fn))['message']
    header, body = d['header'], d['body']

    status_code = header['status_code']
    lyrics_text = []
    tid = basename(fn).split('.json')[0]

    if status_code == 200.:
        if body['lyrics']:
            lyrics = body['lyrics']['lyrics_body'].lower()
            if lyrics != '':
                lyrics_text = [
                    l for l in lyrics.split('\n') if l != ''
                ][:-3]

    return tid, ' '.join(lyrics_text)


def load_lyrics_db(path, fmt='json', verbose=True):
    """ Load loyrics db (crawled) into memory

    Inputs:
        path (string): path where all the api responses are stored
        fmt (string): format of which lyrics are stored
        verbose (bool): indicates whether progress is displayed

    Returns:
        list of tuple: lyrics data
    """
    db = [
        load_mxm_lyrics(fn)
        for fn in tqdm(
            glob.glob(join(path, '*.{}'.format(fmt))),
            disable=not verbose, ncols=80
        )
    ]
    return [(tid, lyrics) for tid, lyrics in db if lyrics != '']
