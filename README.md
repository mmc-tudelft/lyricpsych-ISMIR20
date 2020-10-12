# "Butter Lyrics Over Hominy Grit": Comparing Audio and Psychology-Based Text Features in MIR Tasks
## Supplementary Materials for LyricPsych ISMIR2020 Submission:


### Project Introduction:
This parent repository and its attached child repositories include documentation and various materials that elaborate on the details listed in our research paper. Included are the datasets that we could share, codebases used for various components, and scripts and notebooks used for analysis. The aim of the paper was to explore the use of two psychology-inspired feature sets and one Natural Language Processing-inspired feature set for use in 3 MIR research tasks (genre classification, auto-tagging, and music recommendation). We used a number of baselines in order to compare the performance of our feature sets: a commonly used dictionary of words from psychology research called [LIWC](https://liwc.wpengine.com/), some purely linguistic features such as the number of common words, as well as a set of audio features ([MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)). We crawled the [musiXmatch](https://www.musixmatch.com/) lyrics database for our raw data, extracted our features, implemented a number of systems for each task, and saved the resulting scores in a dataframe for analysis. This initial exploratory study is part of a larger project, whose results will be elaborated as we progress.  


### Paper Abstract:
Psychology research has shown that song lyrics are a rich source of data, yet they are often overlooked in the field of MIR compared to audio. In this paper, we provide an initial assessment of the usefulness of features drawn from lyrics for various fields, such as MIR and Music Psychology. To do so, we asses the performance of lyric-based text features on 3 MIR tasks, in comparison to audio features. Specifically, we draw sets of text features from the field of Natural Language Processing and Psychology. Further, we estimate their effect on performance while statistically controlling for the effect of audio features, by using a hierarchical regression statistical model. Lyric-based features show a small but statistically significant effect, that anticipates further research. Implications and directions for future studies are discussed. 

### Sub Modules: 

This study is largely conducted in 4 steps: `data collection`, `feature extraction`, `task run simulation`, `analysis`. Each sub-module that is lined in this parent repository corresponds to each of those steps:

- `0_mxm_lyrics_crawl`: a simple data crawling utility using musiXmatch SDK
- `1_feature_extraction`: feature extraction package (under development) that is used for this study
- `2_task_run`: contains a codebase for executing the experimental run mimicking the ML system development for 3 MIR tasks, using the feature extracted from the above step.
- `3_analysis`: provides the `R` notebook and relevant data that is used for the analysis of the result collected from the step above.

Most of the functionalities are implemented within `python3` and the analysis is conducted in `R`. The technical details of each step can be found in each sub-repositorie.


### Getting Started:

You can clone the repository and pull the submodules by calling:

```
$git submodule update --init --recursive
```

It takes a few minutes to download the git history, it will eventually download about 800Mb of data.


### Reference and Citation:

TBD
