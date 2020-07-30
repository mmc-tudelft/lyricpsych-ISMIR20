from collections import OrderedDict
import json

from tqdm import tqdm
import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

from .files import personality_adj, value_words, liwc_dict, mxm2msd
from .preprocessing import preprocessing, filter_english_plsa


class Corpus:
    def __init__(self, ids, texts, filt_non_eng=True,
                 filter_stopwords=True, filter_thresh=[5, .3]):
        """"""
        self.ids = ids
        self.texts = texts
        self.filt_non_eng = filt_non_eng
        self.filter_stopwords = filter_stopwords
        self.filter_thresh = filter_thresh

        if filt_non_eng:
            self.ids, self.texts = tuple(zip(
                *filter_english_plsa(
                    list(zip(self.ids, self.texts)),
                    preproc=(
                        False if filter_thresh is None
                        else filter_thresh
                    )
                )
            ))
        self._preproc()

    def _preproc(self):
        output = preprocessing(
            self.texts, 'unigram',
            self.filter_thresh, self.filter_stopwords
        )
        self.ngram_corpus = output[0]
        self.corpus = output[1]
        self.id2word = output[2]
        self.doc_term = output[3]


def load_mxm2msd():
    """ Load the id-map between MxM and MSD

    Inputs:
        fn (str): filename

    Returns:
        dict[str] -> str: MxM to MSD tid
    """
    res = {}
    with open(mxm2msd()) as f:
        for line in f:
            mxm, msd = line.strip().split(',')
            res[mxm] = msd
    return res


def load_personality_adj():
    """ Load personality adjective from Saucier, Goldbberg 1996

    Returns:
        dict[string] -> list of strings: personality adjectives
    """
    return json.load(open(personality_adj()))


def load_value_words():
    """ Load value words from Wilson et al. 2018

    Returns:
        dict[string] -> list of strings: value words
    """
    return json.load(open(value_words()))


def load_liwc_dict():
    """ Load value LIWC dictionary

    Returns:
        dict[string] -> list of strings: value words
    """
    liwc_fn = liwc_dict()
    if liwc_fn is None:
        return None

    return json.load(open(liwc_fn), object_pairs_hook=OrderedDict)
