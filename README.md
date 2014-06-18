langid
======

A demo version of a language identifier using trigram statistics with Katz-Backoff smoothing.

The training files are the respective Wikipedia articles 'dog' for each language.
If the article was too short, I also added another the article about the country's capital.

## Requirements

Linux shell and Python2.7 with standard modules should be enough. 

## Training mode

Langid has two modes: train and test. Test is the default.

There are no trained language models supplied in this repo, so you need to train them at first.
A default training would look like this

    mkdir model
    python identify_lang.py --mode train

These commands will train the language models using the training files found in the directory `train`.



## Upcoming features

1. Reading from stdin and identifying language for each line.
2. Change texts and filenames to English from Hungarian.
