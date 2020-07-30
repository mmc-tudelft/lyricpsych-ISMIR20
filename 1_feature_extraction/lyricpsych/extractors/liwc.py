import logging
from collections import Counter
from itertools import chain

import pandas as pd

from ..data import load_liwc_dict
from .base import BaseTextFeatureExtractor, TextFeature


class LIWC(BaseTextFeatureExtractor):
    def __init__(self):
        super().__init__()

        self.liwc_dict = {}
        self._raw_liwc_dict = load_liwc_dict()
        if self._raw_liwc_dict is None:
            logging.warning('No LIWC dictionary found.'
                            'LIWC feature is not extracted')
        else:
            self._liwc_cat_map = {k:i for i, k
                                  in enumerate(self._raw_liwc_dict.keys())}
            for cat, words in self._raw_liwc_dict.items():
                for word in words:
                    if word not in self.liwc_dict:
                        self.liwc_dict[word] = []
                    self.liwc_dict[word].append(cat)

            # learn the trie
            self._liwc_trie = LIWC._build_liwc_trie(self.liwc_dict)

    @staticmethod
    def _build_liwc_trie(liwc_dict):
        """
        Build a character-trie from the plain string value -> key map

        it is adopted from the `https://github.com/chbrown/liwc-python/blob/master/liwc/trie.py

        Inputs:
            dict[str] -> str: word -> cat map

        Outputs:
            dict[char] -> dict: trie for LIWC
        """
        trie = {}
        for word, cat in liwc_dict.items():
            cursor = trie
            for char in word:
                if char == "*":
                    cursor["*"] = cat
                    break
                if char not in cursor:
                    cursor[char] = {}
                cursor = cursor[char]
            cursor["$"] = cat
        return trie

    @staticmethod
    def _search_trie(trie, token, token_i=0):
        """
        Search the given character-trie for paths match the `token` string

        it is adopted from the `https://github.com/chbrown/liwc-python/blob/master/liwc/trie.py
        """
        if "*" in trie:
            return trie["*"]
        if "$" in trie and token_i == len(token):
            return trie["$"]
        if token_i < len(token):
            char = token[token_i]
            if char in trie:
                return (
                   LIWC._search_trie(trie[char], token, token_i+1)
                )
        return []

    @property
    def is_liwc(self):
        return hasattr(self, '_liwc_trie')

    def extract(self, corpus):
        """
        TODO: deal with special cases (i.e. `(i) like*`)
        """
        if not self.is_liwc:
            raise ValueError('[ERROR] LIWC is not loaded!')

        feats = []
        for words in corpus.ngram_corpus:
            # extract liwc registers
            cnt = Counter(chain.from_iterable([
                LIWC._search_trie(self._liwc_trie, word)
                for word in words
            ]))
            feats.append(dict(cnt))

        # convert to the dataframe -> text feature
        feats = pd.DataFrame(feats).fillna(0.)
        feats = TextFeature('LIWC', corpus.ids, feats.values, feats.columns)
        return feats
