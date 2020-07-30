from itertools import chain

import numpy as np
import pandas as pd

from tqdm import tqdm

from ..preprocessing import SW as stop_words
from .base import BaseTextFeatureExtractor, TextFeature


class LinguisticFeature(BaseTextFeatureExtractor):
    def __init__(self, extreme_thresh=[2, .75]):
        super().__init__()

        self.extreme_thresh = extreme_thresh

    def extract(self, corpus, verbose=False):
        """
        """
        feature = _compute_linguistic_features(corpus.ngram_corpus, verbose)
        feature = feature.dropna(axis=1)
        return TextFeature(
            'linguistic', corpus.ids, feature.values, feature.columns
        )


def _compute_linguistic_features(words_corpus, show_progress=True,
                                 extreme_thresh=[2, 0.75]):
    """ Compute all the linguistic features

    Inputs:
        words_corpus (list of list of string):

    """
    # pre-compute some entities
    doc_freq = get_document_frequency(words_corpus)
    rare_words, common_words = get_extreme_words(doc_freq, extreme_thresh)

    # compute all features
    feats = []
    with tqdm(total=len(words_corpus), ncols=80,
              disable=not show_progress) as p:

        for i, words in enumerate(words_corpus):
            feat = {}
            feat['num_words'] = N = _num_words(words)
            if N is not None:
                # feat['num_rep_phrase'] = _num_rep_phrase(words, phrase_dict)
                feat['num_unique_words'] = _num_unique_words(words)
                feat['num_stop_words'] = _num_stop_words(words)
                feat['num_rare_words'] = _num_extreme_words(words, rare_words)
                feat['num_common_words'] = _num_extreme_words(words, common_words)
                feat['ratio_unique_words'] = feat['num_unique_words'] / N
                feat['ratio_stop_words'] = feat['num_stop_words'] / N
                feat['ratio_rare_words'] = feat['num_rare_words'] / N
                feat['ratio_common_words'] = feat['num_common_words'] / N
            else:
                feat['num_unique_words'] = 0
                feat['num_stop_words'] = 0
                feat['num_rare_words'] = 0
                feat['num_common_words'] = 0
                feat['ratio_unique_words'] = None
                feat['ratio_stop_words'] = None
                feat['ratio_rare_words'] = None
                feat['ratio_common_words'] = None
            feats.append(feat)
            p.update(1)
    feats = pd.DataFrame(feats)
    return feats


def words_sanity_check(ling_feat):
    def wrapper(words, *args, **kwargs):
        # series of sanity checks
        if len(words) == 0:
            # raise Exception('[ERROR] No lyrics found!')
            return None
        else:
            return ling_feat(words, *args, **kwargs)
    return wrapper


@words_sanity_check
def _num_words(words):
    """ Count number of words per songs
    """
    return len(words)


@words_sanity_check
def _num_rep_phrase(words, phrase_dict):
    """ count appearance of phrase give in the phrase dict
    """
    pass


@words_sanity_check
def _num_unique_words(words):
    """ Count unique number of words per song
    """
    return len(set(words))


@words_sanity_check
def _num_stop_words(words):
    """ Count the number of stop words included
    """
    return len([w for w in set(words) if w in stop_words])


@words_sanity_check
def _num_extreme_words(words, extreme_words, average=True):
    """ Count the number of common words

    Inputs:
        words (list of string): to be checked
        extreme_words (set of string or dict[string] -> float): common words set

    Returns:
        tuple or list of int: # of extreme words in each extreme polars
    """
    if not isinstance(extreme_words, (dict, set)):
        raise Exception('[ERROR] common/rare word list should be set!')
    elif isinstance(extreme_words, list):
        extreme_words = set(extreme_words)

    if not len(extreme_words) > 0:
        raise Exception('[ERROR] no words found!!')

    res = 0
    for word in words:
        if word in extreme_words:
            res += 1

    if average:
        res /= len(extreme_words)

    return res


def get_extreme_words(df, thresh=[2, .95]):
    """ Extract extreme words in each polars

    Inputs:
        idf (dict[string] -> float): contains words and
                                     their document raw frequency (in count)
        thresh ([int, float]): threshold to determine extreme words.
                               the first element is the threshold for rare words.
                               words appeared less than this number are treated as rare
                               the second element is the threshold for common words.
                               words appeared more than this ratio (to the entire corpus)
                               considered as the common word.

    Returns:
        tuple of string: extreme words.
    """
    df_arr = np.array(list(df.values()))
    com_thrs = np.percentile(df_arr, thresh[1] * 100)

    rar_words = set(w for w, freq in df.items() if freq < thresh[0])
    com_words = set(w for w, freq in df.items() if freq > com_thrs)
    return rar_words, com_words


def get_document_frequency(corpus):
    """ Get document frequency from given corpus

    Inputs:
        texts (list of list of string): corpus

    Returns:
        dict[string] -> float: list of vocabulary and their document frequency
    """
    unique_words = set(chain.from_iterable(corpus))
    df = dict.fromkeys(unique_words, 0)
    for words in corpus:
        for word in words:
            df[word] += 1
    return df
