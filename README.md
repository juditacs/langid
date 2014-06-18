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

These commands will train the language models using the training files found in the directory `train` (`--train-files` option).
By deafult there are train files for 7 languages (Dutch, English, French, German, Hungarian, Italian, Spanish)
but you're free to add more or use different ones.
The trained models are in the `model` directory (`--model` option).

## Language detecting mode

Once the models are trained, you can use them to identify the language of the input.
There are two kinds of input supported:

1. STDIN: each line of the standard input is treated as a separate document
2. Document-level: a language is assigned to each document in a directory

The output is a tab-separated file with the following columns:

1. Input line from STDIN (type 1 input) or the path of the document (type 2 input)
2. Most probable language
3. Log-probability of the language
4. 2nd most probable language
5. Log-probability of the language

...

At most 5 languages are assigned to one line of input or one document.

## Options

| `-N` | N in ngram | 3 |

## Mathematical correctness


