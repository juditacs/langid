import argparse
import itertools
import math
from collections import defaultdict
from sys import stderr, stdin
from os import path, listdir
import re


whitespace_re = re.compile(r'\s+', re.UNICODE)


def identify_input(models):
    if args.test_dir:
        for fn in listdir(args.test_dir):
            with open(path.join(args.test_dir, fn)) as f:
                clean = clean_text(f.read().decode('utf8', 'ignore'))[:args.test_cutoff]
                probs, st = compute_probabilities(clean, models)
                print(fn + '\t' + '\t'.join('{0}\t{1}'.format(lang, prob) for lang, prob in sorted(probs.iteritems(), key=lambda x: -x[1])[0:5] if prob > float('-inf') and prob < 0))
    else:
        for l in stdin:
            clean = clean_text(l.decode('utf8', 'ignore'))[:args.test_cutoff]
            probs, st = compute_probabilities(clean, models)
            print(clean.encode('utf8') + '\t' + '\t'.join('{0}\t{1}'.format(lang, prob) for lang, prob in sorted(probs.iteritems(), key=lambda x: -x[1])[0:5] if prob > float('-inf') and prob < 0))


def compute_probabilities(input_str, models):
    input_ngrams = get_seen_ngrams(input_str)
    probs = defaultdict(lambda: 0.0)
    stats = defaultdict(lambda: [0, 0])
    for lang, model in models.iteritems():
        for ngram in input_ngrams:
            if len(ngram) != args.N:
                continue
            stats[lang][0] += 1
            if ngram in model:
                probs[lang] += model[ngram]
            else:
                stats[lang][1] += 1
                if args.verbose:
                    stderr.write('Ngram {0} not found in {1} model\n'.format(ngram, lang))
                #probs[lang] += float('-inf')
    return probs, stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-N', type=int, default=3)
    p.add_argument('-c', '--cutoff', type=int, default=1000)
    p.add_argument('--test-cutoff', type=int, default=100)
    p.add_argument('-d', '--discount', type=float, default=0.5)
    p.add_argument('-m', '--mode', default='test', choices=['train', 'test'])
    p.add_argument('--train-files', default='train_files', type=str)
    p.add_argument('--test-dir', type=str)
    p.add_argument('--models', default='models', dest='model_dir', type=str)
    p.add_argument('-l', '--lower', action='store_true', default=False)
    p.add_argument('-v', '--verbose', action='store_true', default=False)
    return p.parse_args()


def clean_text(text):
    if args.lower:
        t = text.lower()
    else:
        t = text
    #t = t.replace('\n', ' ')
    #t = t.replace('\t', ' ')
    t = whitespace_re.sub(' ', t)
    return t.strip()


def train_models():
    stderr.write('Training models...\n')
    # reading train_files
    for lang in listdir(args.train_files):
        stderr.write(lang + '\n')
        with open(path.join(args.train_files, lang)) as f:
            text = f.read().decode('utf8')[0:args.cutoff]
            text_clean = clean_text(text)
        ngrams = get_seen_ngrams(text_clean)
        model = get_probabilities(ngrams)
        write_model(model, lang)


def get_probabilities(ngrams):
    probs = defaultdict(lambda: float('-inf'))
    compute_unigram_probs(probs, ngrams)
    alphabet = set(probs.keys())
    for n in range(2, args.N + 1):
        compute_katz_probs(probs, ngrams, n, alphabet)
    return probs


def compute_katz_probs(probs, ngrams, n, alphabet):
    discount = args.discount
    leftover = defaultdict(float)
    # iterate seen ngrams and compute probability
    for ngram, count in ngrams.iteritems():
        if len(ngram) != n:
            continue
        if ngram in probs:
            print('BAJ VAN')
        probs[ngram] = math.log((float(count) - discount) / ngrams[ngram[:-1]])
        leftover[ngram[:-1]] += (float(count) - discount) / ngrams[ngram[:-1]]
    missing = defaultdict(float)
    # probability of every other ngram
    for ngram_ in itertools.product(alphabet, repeat=n):
        ngram = ''.join(ngram_)
        if ngram in probs:
            continue
        missing[ngram[:-1]] += math.exp(probs[ngram[1:]])
    for ngram_ in itertools.product(alphabet, repeat=n):
        ngram = ''.join(ngram_)
        if ngram in probs:
            continue
        try:
            probs[ngram] = math.log((1 - leftover[ngram[:-1]])) + probs[ngram[1:]] - math.log(missing[ngram[:-1]])
        except ValueError:
            probs[ngram] = float('-inf')


def compute_unigram_probs(probs, ngrams):
    unigram_sum = sum((v for k, v in ngrams.iteritems() if len(k) == 1))
    for ngram, count in ngrams.iteritems():
        # skip if not a unigram
        if len(ngram) > 1:
            continue
        probs[ngram] = math.log(float(count) / unigram_sum)


def get_seen_ngrams(text):
    ngrams = defaultdict(int)
    for n in range(1, args.N + 1):
        for i in range(0, len(text) - n + 1):
            ngram = text[i:i + n]
            ngrams[ngram] += 1
    return ngrams


def write_model(model, lang):
    with open(path.join(args.model_dir, lang + '.model'), 'w') as f:
        f.write('\n'.join(u'{0}\t{1}'.format(ngram, prob) for ngram, prob in sorted(model.iteritems(), key=lambda x: -x[1])).encode('utf8') + '\n')
        stderr.write('Model written to file: {0}\n'.format(path.join(args.model_dir, lang + '.model')))


def write_models(models):
    for lang, model in models.iteritems():
        with open(path.join(args.model_dir, lang + '.model'), 'w') as f:
            f.write('\n'.join(u'{0}\t{1}'.format(ngram, prob) for ngram, prob in sorted(model.iteritems(), key=lambda x: -x[1])).encode('utf8') + '\n')


def read_models():
    models = defaultdict(lambda: defaultdict(float))
    for fn in listdir(args.model_dir):
        lang = fn.rstrip('model')[:-1]
        stderr.write('Reading {0}\n'.format(fn))
        with open(path.join(args.model_dir, fn)) as f:
            for l in f:
                ngram, prob = l.decode('utf8').split('\t')
                models[lang][ngram] = float(prob)
    return models

args = parse_args()


def main():
    if args.mode == 'train':
        train_models()
        #write_models(models)
    else:
        models = read_models()
        identify_input(models)

if __name__ == '__main__':
    main()
