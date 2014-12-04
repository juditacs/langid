langid
======

A simple language identifier using trigram statistics with Katz's back-off smoothing.

The training files are the respective Wikipedia articles 'dog' for each language.
If the article was too short, I also added another the article about the country's capital.

## Requirements

Linux shell and Python2.7 with standard modules should be enough. 
You may need a few Gigabytes of memory for many languages or very large training files.

## Training mode

Langid has two modes: train and test (`--mode` option). Test is the default.

There are no trained language models supplied in this repo, so you need to train them at first.
A default training would look like this

    mkdir model
    python identify_lang.py --mode train

These commands will train the language models using the training files found in the directory `train` (`--train-files` option).
By deafult there are train files for 7 languages (Dutch, English, French, German, Hungarian, Italian, Spanish)
but you're free to add more or use different ones.
The trained models are in the `model` directory (`--model-files` option).
They are named as the training files with the suffix `.model`.

## Language detecting mode (test mode)

Once the models are trained, you can use them to identify the language of the input.
There are two kinds of input supported:

1. STDIN: each line of the standard input is treated as a separate document. This is the default.
2. Document-level: a language is assigned to each document in a directory (this mode is used if the `--test-files` option is specified).

The output is a tab-separated file with the following columns:

1. Input line from STDIN (type 1 input) or the path of the document (type 2 input)
2. Most probable language
3. Log-probability of the language
4. 2nd most probable language
5. Log-probability of the language

...

At most 5 languages are assigned to one line of input or one document.

### Examples

Reading from STDIN:

    echo "black cats and dogs" | python identify_lang.py > results

Identify each document in a directory

    mkdir test
    echo "black cats and dogs" > test/en
    echo "fekete kutyák és macskák" > test/hu
    python identify_lang.py --test-files test > results

## Options

| Option  | Explanation | Default |
| ------------- | ------------- | --- |
| `-N` | N in ngram | 3 |
| `-c` | Train cutoff: use first c characters of each train file | 10,000 |
| `--test-cutoff` | Test cutoff: use first c characters of each identifiable file. You may use this to avoid very low probabilities in case of very large documents | 100 |
| `-l`, `--lower` | Lowercase all input. If specified, make sure that the models were also built with this flag. | False |
| `-d` | Discount parameter of Katz's back-off. If changing it from 0.5 helps, please let me know. | 0.5 |
| `-m`, `--mode` | Train or test mode | test |
| `--train-files` | Directory containing the train files | `train` |
| `--test-files` | Directory containing the test files. STDIN is read if this option is not specified | None |
| `--model-files` | Directory of the model files. Models are written to this directory in training mode and are read from this directory in testing mode. | `model` |
| `-v`, `--verbose` | Verbose output. See below | False |

## Mathematical correctness

Mathematical correctness suffers for two reasons:

1. Initial and ending probabilities are ignored because of the lack of document level training data.
2. Unseen characters (and therefore unseen trigrams) are counted but the probabilities are not nuked (they would be zero or -inf in log-prob in this case).

### Verbose output

Seen and unseen trigrams are counted and printed to the verbose output. If the verbose output is used, the columns change slightly:

1. Input line from STDIN (type 1 input) or the path of the document (type 2 input)
2. Most probable language
3. Log-probability of the language
4. Number of trigrams appearing in the language model
5. Number of trigrams not appearing in the language model
4. 2nd most probable language
5. Log-probability of the language
4. Number of trigrams appearing in the language model
5. Number of trigrams not appearing in the language model

...

## Contact

Please send your feedback, questions and bugreports to judit@sch.bme.hu

