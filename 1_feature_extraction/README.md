# LyricPsych: towards a psychologically validated feature extractors for the lyrics data

This package contains a number of feature extraction methods on textual data, mostly expecting musical lyrics. As an work-in-progress development phase, we will continuously validate and update the package later on. Currently, several feature extraction methods including `linguistic features`, `topic_modeling`, `psychology inventory based feature estimations`, etc.


## Getting Started

To install this package, we recommend you use the python virtual environment. Inside the virtualenv, installation is using `pip` and `git`.

```console
$ pip install git+https://github.com/mmc-tudelft/lyricpsych.git@packaging
```
feature extractor `lyricpsych-extract` installed along with the package. The usage of the `lyricpsych-extractor` can be found in the `-h` option. For instance, you can extract `personality`, `value`, `topic`, `linguistic` features by using the example below:

```console
$ lyricpsych-extract \
    /path/to/the/lyrics_data.csv \
    /path/for/output/ \
    --w2v glove-twitter-25 \
    --features linguistic value personality topic
```


## Contributing

Currently it's in its alpha version. It means some extractors are not fully validated, and may have a unexpected behavior. We will continue to work on improving those aspects, but also we are more than welcoming contributions. We are open to take issues and pull request.


## TODO list

- [x] refactoring
  - [x] split extractor to dedicated extractors
  - [x] minor refactorings
  - [x] clean up
    - [x] unused functions
    - [x] unused data files
    - [x] unused scripts
  - [x] restructuring
    - [x] split `task` to the separate sub-module
    - [x] separate `fm` and `als_feat` to the separate repo
- [ ] Documentation
  - [ ] docstrings
  - [ ] doc generation
- [x] features
  - [x] experimental run reproduction cli
- [ ] deploy
  - [ ] writing testings
  - [ ] CI [Travis integration]
  - [ ] register to PyPI


## Authors

Jaehun Kim, Sandy Manolios

## Reference

TBD

## Acknowledgement

- Thanks to [henryre](https://github.com/henryre), as our PLSA implementation of this API is extension of [numba-plsa](https://github.com/henryre/numba-plsa).
