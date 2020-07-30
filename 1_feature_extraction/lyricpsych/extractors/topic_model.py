import os

# TODO: generalize and make this more adaptive/interactive following steps
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '2'

import numpy as np
import numba as nb
from tqdm import tqdm

from .base import BaseTextFeatureExtractor, TopicFeature


class TopicModel(BaseTextFeatureExtractor):
    def __init__(self, k=25):
        super().__init__()

        self.k = k
        self._topic_model = PLSA(k, n_iters=30)

    def extract(self, corpus):
        """
        """
        self._topic_model.fit(corpus.doc_term)
        return TopicFeature(
            self.k, corpus.ids, self._topic_model.doc_topic,
            self._topic_model.topic_term, corpus.id2word.token2id
        )


class PLSA:
    def __init__(self, k, n_iters=30):
        self.k = k
        self.n_iters = n_iters

    def fit(self, X):
        coo = X.tocoo()
        theta, beta = self._init_params(coo)

        plsa_numba(
            coo.row, coo.col, coo.data,
            theta, beta, self.n_iters
        )

        # assign
        self.doc_topic = theta
        self.topic_term = beta

    def _init_params(self, X, init_theta=True, init_beta=True):
        theta, beta = None, None

        if init_theta:
            theta = np.random.rand(X.shape[0], self.k)
            theta = theta / theta.sum(1)[:, None]
            theta = theta.astype(np.float32)

        if init_beta:
            beta = np.random.rand(X.shape[1], self.k).T
            beta = beta / beta.sum(0)[None]
            beta = beta.astype(np.float32)

        return theta, beta

    def transform(self, X):
        if not hasattr(self, 'topic_term'):
            raise Exception('[ERROR] .fit should be called before!')

        X = X.tocoo()
        theta = self._init_params(X, init_beta=False)[0]

        _learn_doc_topic(
            X.row, X.col, X.data,
            theta, self.topic_term, self.n_iters
        )
        return theta

    def score(self, X):
        return _perplexity(X, self, self.n_iters)


@nb.njit(parallel=True, nogil=True, fastmath=True)
def plsa_numba(dt_row, dt_col, dt_val, theta, beta, n_iter, eps=1e-10):
    """ PLSA numba. this function is extension from:

    https://github.com/henryre/numba-plsa

    This version is not inflating full phi term (nnz * K memory), but using
    ((M + N) * K), which is typically << (nnz * K). it could even be improved
    by the way used in gensim's LDA implementation which does not use any
    additional temporary container for update.

    Also, online update similar to OnlineLDA could be speed up the fitting further.
    """
    M, K = theta.shape
    V = beta.shape[1]

    nnz = len(dt_val)
    dtype = theta.dtype
    # phi_sum = np.empty((nnz,), dtype=dtype)
    beta_sum = np.empty((K,), dtype=dtype)
    theta_sum = np.empty((M,), dtype=dtype)

    # for update
    theta_next = np.empty((M, K), dtype=dtype)
    beta_next = np.empty((K, V), dtype=dtype)

    rnd_idx = np.random.permutation(nnz)
    for i in range(n_iter):
        theta_next[:] = 0.
        beta_next[:] = 0.
        theta_sum[:] = eps
        beta_sum[:] = eps
        for ii in nb.prange(nnz):
            idx = rnd_idx[ii]
            d, t = dt_row[idx], dt_col[idx]
            s = theta.dtype.type(0.)
            phi = np.zeros((K,), dtype=dtype)
            for z in range(K):
                phi[z] = theta[d, z] * beta[z, t]
                s += phi[z]
            if s == 0:
                s = eps

            for z in range(K):
                q = dt_val[idx] * phi[z] / s
                beta_next[z, t] += q
                beta_sum[z] += q
                theta_next[d, z] += q
                theta_sum[d] += q

        # normalize P(topic | doc)
        for d in range(M):
            for z in range(K):
                theta[d, z] = theta_next[d, z] / theta_sum[d]

        # normalize P(term | topic)
        for t in range(V):
            for z in range(K):
                beta[z, t] = beta_next[z, t] / beta_sum[z]


@nb.njit
def _learn_doc_topic(dt_row, dt_col, dt_val, theta, beta, n_iter, eps=1e-10):
    """ PLSA numba. this function is directly employed from:

    https://github.com/henryre/numba-plsa
    """
    M, K = theta.shape
    V = beta.shape[1]

    nnz = len(dt_val)
    dtype = theta.dtype
    theta_sum = np.empty((M,), dtype=dtype)

    # for update
    theta_next = np.empty((M, K), dtype=dtype)

    rnd_idx = np.random.permutation(nnz)
    for i in range(n_iter):
        theta_next[:] = 0.
        theta_sum[:] = eps
        for ii in nb.prange(nnz):
            idx = rnd_idx[ii]
            d, t = dt_row[idx], dt_col[idx]
            s = theta.dtype.type(0.)
            phi = np.zeros((K,), dtype=dtype)
            for z in range(K):
                phi[z] = theta[d, z] * beta[z, t]
                s += phi[z]
            if s == 0:
                s = eps

            for z in range(K):
                q = dt_val[idx] * phi[z] / s
                theta_next[d, z] += q
                theta_sum[d] += q

        # normalize P(topic | doc)
        for d in range(M):
            for z in range(K):
                theta[d, z] = theta_next[d, z] / theta_sum[d]

def _perplexity(test_docs, plsa, n_iter=30):
    """ Compute perplexity of given topic model

    Inputs:
        test_docs (scipy.sparse.csr_matrix): hold-out document
        plsa (PLSA): trained pLSA model

    Returns:
        float: perplexity
    """
    test_docs = test_docs.tocsr()
    new_theta = plsa.transform(test_docs)
    log_p_w_theta = np.zeros(test_docs.shape[1])
    for doc_idx in range(test_docs.shape[0]):
        phi = new_theta[doc_idx][None]
        internal_idx = slice(
            test_docs.indptr[doc_idx],
            test_docs.indptr[doc_idx+1]
        )
        idx = test_docs.indices[internal_idx]
        val = test_docs.data[internal_idx]

        log_p_w_theta[idx] += (
            np.log(np.maximum(phi @ plsa.topic_term[:, idx], 1e-14))[0] * val
        )

    perplexity = np.exp(-log_p_w_theta.sum() / test_docs.sum())
    return perplexity


def get_top_terms(plsa, id2word, topk=20):
    return [
        [
            id2word[t] for t
            in np.argsort(-plsa.topic_term[kk])[:topk]
        ]
        for kk in range(plsa.k)
    ]
