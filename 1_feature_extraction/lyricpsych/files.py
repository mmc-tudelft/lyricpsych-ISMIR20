import os
import pkg_resources

LIWC = os.environ.get('LIWC_PATH')  # this is required to be set
PERS_ADJ = 'data/personality_adjectives.json'
VAL_WORDS = 'data/value_inventory_Wilson18.json'
MXM2MSD = 'data/mxm2msd.txt'
# they're generated randomly from https://www.song-lyrics-generator.org.uk/
TEST_LYRICS = 'data/test_lyrics.txt'


__all__ = [
    'personality_adj',
    'value_words',
    'liwc_dict',
    'mxm2msd',
    'test_lyrics'
]


def personality_adj():
    """ Read the filename of personality

    Returns:
        str: filename of personality adjectives
    """
    return pkg_resources.resource_filename(__name__, PERS_ADJ)


def value_words():
    """ Read the filename of values

    Returns:
        str: filename of value words
    """
    return pkg_resources.resource_filename(__name__, VAL_WORDS)


def liwc_dict():
    """ Read the filename of LIWC dictionary

    Returns:
        str: filename of LIWC dictionary
    """
    if LIWC is None:
        return None
    return LIWC


def mxm2msd():
    """ Read the filename of map between MxM and MSD

    Returns:
        str: filename of map between MxM and MSD
    """
    return pkg_resources.resource_filename(__name__, MXM2MSD)


def test_lyrics():
    """ Read the filename of the testing lyrics

    .. note::
        the lyrics are randomly generated from the website:
            `Song Lyrics Generator <https://www.song-lyrics-generator.org.uk/>`__

    Returns:
        str: filename of testing lyrics
    """
    return pkg_resources.resource_filename(__name__, TEST_LYRICS)
