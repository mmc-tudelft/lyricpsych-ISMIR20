from os.path import exists, basename, splitext
import json

import numpy as np
import gensim
import gensim.downloader
import jsonschema

from .base import BaseTextFeatureExtractor, TextFeature
from ..data import load_personality_adj, load_value_words
from ..utils import normalize_matrix

"""
TODO: propper logging
"""


INVENTORY_JSONSCHEMA = {
    "type":"object",
    "patternProperties": {
        # it takes any field name
        "^.*$": {
            # only allows array (list in python) as the value
            "type":"array",
            # only allows string as the item within the array
            "items":{"type":"string"}
        }
    }
}


class InventoryScore(BaseTextFeatureExtractor):
    def __init__(self, inventory=None, gensim_w2v=None,
                 compute_similarity=False):
        """
        """
        super().__init__()

        # this can take minutes if the model if big one
        self.w2v = gensim_w2v
        self.inventory = inventory
        self.compute_similarity = compute_similarity
        self.inventory, self.inventory_name = self._load_inventory(inventory)

        # pre-compute the inventory
        self.grps, self.grp_embs = self._compute_inventory_avg_emb()

    @staticmethod
    def _load_inventory(inventory, use_filename=True):
        """
        """
        if inventory is None:
            return None, ''
        elif inventory == 'personality':
            return load_personality_adj(), inventory
        elif inventory == 'value':
            return load_value_words(), inventory
        elif isinstance(inventory, dict):
            # check the schema using the jsonschema
            jsonschema.validate(INVENTORY_JSONSCHEMA, inventory)
            return inventory, 'custom_inventory'
        else:
            if not exists(inventory):
                raise ValueError('[ERROR] cannot find inventory file {}!'
                                 .format(inventory))

            if use_filename:
                name = splitext(basename(inventory))[0]
            else:
                name = 'custom_inventory'

            # load custom dictionary
            inv = json.load(open(inventory))

            # check if the inventory well following the format
            # TODO: if there's more validation step needed, should do it
            jsonschema.validate(INVENTORY_JSONSCHEMA, inv)

            return inv, name

    def _compute_inventory_avg_emb(self):
        """
        """
        ids, embs = [], []
        for grp, wrds in self.inventory.items():
            # register group ids to the container
            ids.append(grp)

            # get embeddings
            embeddings = [self.w2v[w] for w in wrds if w in self.w2v]
            if len(embeddings) == 0:
                embeddings = np.zeros((self.w2v.vector_size,))
            else:
                embeddings = np.array(embeddings)
            embeddings = normalize_matrix(embeddings, axis=1)

            # register average embeddings
            embs.append(embeddings.mean(0))
        return ids, np.array(embs)

    def extract(self, corpus):
        """
        """
        if self.w2v is None:
            raise ValueError('[ERROR] word embedding model is not loaded!')

        doc_embs = []
        for wrds in corpus.ngram_corpus:
            embs = [self.w2v[w] for w in wrds if w in self.w2v]
            if len(embs) == 0:
                # print('[ERROR] found there is no words matched with W2V model!')
                doc_embs.append(np.zeros((self.w2v.vector_size,)))
            else:
                embs = normalize_matrix(np.array(embs), axis=1)
                doc_embs.append(embs.mean(0))
        doc_embs = np.array(doc_embs)

        # compute the score [cosine similarity]
        scores = doc_embs @ self.grp_embs.T
        if not self.compute_similarity:
            scores = 1 - scores

        # convert to data to text feature
        feats = TextFeature(
            self.inventory_name, corpus.ids, scores, self.grps
        )
        return feats


def load_word_embedding(w2v):
    """ Load the load embedding model

    Currently, it depends on `gensim` w2v model only.
    TODO: custom w2v model with native `dict`

    Inputs:
        w2v (string): name of the gensim w2v model

    Outputs:
        gensim.models.Word2Vec or dict
    """
    return gensim.downloader.load(w2v)
