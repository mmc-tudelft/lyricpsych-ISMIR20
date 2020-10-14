# Musixmatch lyrics client using Musixmatch API 

This small utility is to help who wants to collect the lyrics data from Musixmatch. It is based on the python API client from [Musixmatch SDK](https://github.com/musixmatch/musixmatch-sdk).

We strongly recommend to use a virtual environment to use this as the Musixmatch API is only supporting deprecated python versions (2.7 / 3.4).

We also expect the potential user of this utility to already have a `API key` from [Musixmatch developer](https://developer.musixmatch.com/).


## Getting Started

Installation is straightforward, simply cloning this repo and run the shell script installing the python SDK wrapper for your environment.

```bash
(py27_venv)$ git clone lyricpsych-mxm-crawler
(py27_venv)$ sh install.sh
```

Once you have the API key for MxM, it's ready to test out the API.


### Quick Look

First you need prepare a text file listing the target songs (Musixmatch track ids). For instance:

```
674743
4280619
…
```

If you want to download a number of songs in bulk, you may want to take a look at the [Million Song Dataset](http://millionsongdataset.com/musixmatch/) which provides the map between song metadata included in the dataset and the Musixmatch track ids.

```bash
(py27_venv)$ export MXM_APIKEY="(your_apikey)"
(py27_venv)$ python get_lyrics.py target_ids.txt --out-dir /path/to/outputs/ --verbose
```
Then you can find the response json files from at `/path/to/outputs/` directory.

## Current Status & Contributing

This is very small piece of utility and not fully tested. Also, it's currently not directly support the windows machines. For windows users, one can still use WSL or Anaconda to use this. We are always open to the contribution in any form.

## Authors

- Jaehun Kim

## License


This project is licensed under the MIT License - see the LICENSE.md file for details


## Reference
