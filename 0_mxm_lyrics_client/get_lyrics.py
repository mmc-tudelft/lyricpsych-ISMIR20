import os
from os.path import join
import json
import argparse

# musixmatch API python interface
import swagger_client as mxm

# for monitoring
from tqdm import tqdm


# get API key from environment variable
if 'MXM_APIKEY' not in os.environ:
    raise ValueError('[ERROR] cannot find MxM API key in the environment'
                     ' variable. (under "MXM_APIKEY")')

# register API key to the client
mxm.configuration.api_key['apikey'] = os.environ['MXM_APIKEY']


# prepare argparser to get the target list
parser = argparse.ArgumentParser()
parser.add_argument('targets', type=str,
                    help=('path to the text file containing'
                          'the target MxM track ids'))
parser.add_argument('--out-dir', type=str, default='./',
                    help='path to save the fetched json files')
parser.add_argument('--verbose', default=False, dest='verbose', action='store_true')
args = parser.parse_args()


# read the targets.
# we expect the text file contains one mxm id per line
#
# i.e.) 
# $ head test.txt
# 123456
# 3573789
# 454136
# 5688335
# ...

with open(args.targets, 'r') as f:
    targets = [l.replace('\n', '') for l in f]

# initialize api instance
api = mxm.LyricsApi()

# fetch each entry
with tqdm(total=len(targets), ncols=80, disable=args.verbose) as prog:
    for mxm_id in targets:
        res = api.track_lyrics_get_get(mxm_id).to_dict()
        json.dump(res, open(join(args.out_dir, '{}.json'.format(mxm_id)), 'w'))
        prog.update()
