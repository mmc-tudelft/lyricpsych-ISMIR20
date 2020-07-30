from os.path import (join,
                     basename,
                     splitext,
                     exists)
import argparse
import logging
import json

from .extractors import (InventoryScore,
                         LinguisticFeature,
                         LIWC,
                         TopicModel)
from .extractors.inventory_score import load_word_embedding
from .data import Corpus
from .utils import save_feature_h5, save_feature_csv


K = 25  # found by internal cross-validation. can be improved
THRESH = [5, 0.3]
EXTENSIONS = {
    'csv': '.csv',
    'hdf5': '.h5'
}
SAVE_FN = {
    'csv': save_feature_csv,
    'hdf5': save_feature_h5
}


class FeatureSetAction(argparse.Action):
    CHOICES = {'linguistic', 'liwc', 'value', 'personality', 'topic'}
    def __call__(self, parser, namespace, values, option_string=None):
        if values:
            for value in values:
                if value not in self.CHOICES:
                    message = ("invalid choice: {0!r} (choose from {1})"
                               .format(value,
                                       ', '.join([repr(action)
                                                  for action in self.CHOICES])))
                    raise argparse.ArgumentError(self, message)
            setattr(namespace, self.dest, values)


def extract_argparse():
    """
    """
    # process the arguments
    parser = argparse.ArgumentParser(
        description="Extracts textual feature using various psych inventories"
    )

    parser.add_argument('text', type=str,
                        help='filename for the file contains text. one sentence per line')

    parser.add_argument('out_path', type=str,
                        help='path where the output (.csv) is stored')

    parser.add_argument('--w2v', default=None, type=str,
                        help='name of the gensim w2v model')

    parser.add_argument('--format', default='csv', type=str,
                        choices=set(EXTENSIONS),
                        help='file format to be saved')

    parser.add_argument('--inventory', dest='inventory', type=str, default=None,
                        help=('filename contains dictionary contains category-words pair'
                              ' that is used for the target inventory'))

    parser.add_argument('--features', nargs="*", action=FeatureSetAction,
                        default=['linguistic', 'personality', 'value', 'topic'],
                        help='indicators for the desired featuresets')

    parser.add_argument('--filt-non-eng', default=False,
                        dest='filt_non_eng', action='store_false')

    parser.add_argument('--k', default=K, type=int,
                        help='the number of topics for topic modeling')

    parser.add_argument('--min-doc-freq', default=THRESH[0], type=int,
                        help='')

    parser.add_argument('--max-doc-freq', default=THRESH[1], type=float,
                        help='')

    return parser.parse_args()


def extract():
    """command line toolchain for feature extraction
    """
    args = extract_argparse()
    flags = set(args.features)
    filter_thresh = [args.min_doc_freq, args.max_doc_freq]

    # setup custom inventory
    if args.inventory is not None:
        if not exists(args.inventory):
            raise ValueError('[ERROR] inventory file not found!')
        else:
            custom_inven = json.load(open(args.inventory))
            flags.add('custom_inventory')

    # 1. loading the data 
    texts = [line.strip() for line in open(args.text)]
    ids = list(range(len(texts)))
    corpus = Corpus(ids, texts, filter_thresh=None,
                    filter_stopwords=False,
                    filt_non_eng=args.filt_non_eng)

    # 2. extract features
    features = {}

    if 'liwc' in flags:
        ext = LIWC()
        features['liwc'] = ext.extract(corpus)

    if 'linguistic' in flags:
        ext = LinguisticFeature()
        features['linguistic'] = ext.extract(corpus)

    need_w2v = {'value', 'personality', 'custom_inventory'}
    if any(flag in need_w2v for flag in flags):
        if args.w2v is not None:
            w2v = load_word_embedding(args.w2v)
        else:
            print('No word2vec model specified! '
                  '`glove-twitter-25` is loaded...')
            w2v = load_word_embedding('glove-twitter-25')

    corpus.filter_thresh = filter_thresh
    corpus.filter_stopwords = True
    corpus._preproc()
    for flag in flags:
        if flag.lower() == 'topic':
            ext = TopicModel()

        elif flag.lower() == 'value':
            ext = InventoryScore('value', w2v)

        elif flag.lower() == 'personality':
            ext = InventoryScore('personality', w2v)

        elif flag.lower() == 'custom_inventory':
            ext = InventoryScore(custom_inven, w2v)

        else:
            continue

        # get features
        features[flag.lower()] = ext.extract(corpus)

    # 3. save the file to the disk
    out_fn = join(
        args.out_path,
        splitext(basename(args.text))[0] + '_feat' + EXTENSIONS[args.format]
    )
    SAVE_FN[args.format](features, out_fn)
