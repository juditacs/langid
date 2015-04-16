#!/usr/bin/env python
import argparse
import itertools
import math
from collections import defaultdict
from sys import stderr, stdin
from os import path, listdir
import re


whitespace_re = re.compile(r'\s+', re.UNICODE)
sentence_re = re.compile(r'([^.?!])+[^.?!]', re.UNICODE)
digit_re = re.compile(r'[0-9]+', re.UNICODE)


def identify_input(models):
    if args.test_files:
        for fn in listdir(args.test_files):
            doc_path = path.join(args.test_files, fn)
            with open(doc_path) as f:
                text = f.read().decode('utf8', 'ignore')[:args.test_cutoff]
                ngrams = get_ngrams_from_text(text, args.token_mode, args.N, padding=True)
                probs, st = compute_probabilities(ngrams, models, args.N)
                output(doc_path, probs, st)
    else:
        for l in stdin:
            text = l.decode('utf8', 'ignore')[:args.test_cutoff].strip()
            ngrams = get_ngrams_from_text(text, args.token_mode, args.N, padding=True)
            probs, st = compute_probabilities(ngrams, models, args.N)
            output(text, probs, st)


def output(prefix, probs, seen):
    if args.verbose:
        print(prefix.encode('utf8') + '\t' +
              '\t'.join(u'{0}\t{1}\t{2}\t{3}'.format(lang, prob, seen[lang][0], seen[lang][1])
                        for lang, prob in
                        sorted(probs.iteritems(), key=lambda x: -x[1])[0:5] if prob > float('-inf')).encode('utf8'))
    else:
        print(prefix.encode('utf8') + '\t' +
              '\t'.join(u'{0}\t{1}'.format(lang, prob)
                        for lang, prob in
                        sorted(probs.iteritems(), key=lambda x: -x[1])[0:5] if prob > float('-inf')).encode('utf8'))


def compute_probabilities(ngrams, models, N):
    probs = defaultdict(lambda: 0.0)
    stats = defaultdict(lambda: [0, 0])
    for lang, model in models.iteritems():
        for ngram in ngrams:
            if len(ngram) != N:
                continue
            stats[lang][0] += 1
            if ngram in model:
                probs[lang] += model[ngram]
            else:
                stats[lang][1] += 1
    return probs, stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-N', type=int, default=3)
    p.add_argument('-c', '--cutoff', type=int, default=10000)
    p.add_argument('--test-cutoff', type=int, default=100)
    p.add_argument('-d', '--discount', type=float, default=0.5)
    p.add_argument('-m', '--mode', default='test', choices=['train', 'test'])
    p.add_argument('--token-mode', default='char', choices=['char', 'word'])
    p.add_argument('--train-files', default='train', type=str)
    p.add_argument('--test-files', type=str)
    p.add_argument('--model-files', default='model', dest='model_dir', type=str)
    p.add_argument('-l', '--lower', action='store_true', default=False)
    p.add_argument('-v', '--verbose', action='store_true', default=False)
    p.add_argument('--languages', dest='filter_langs', default='', type=str)
    return p.parse_args()


def clean_text(text):
    if args.lower:
        t = text.lower()
    else:
        t = text
    t = whitespace_re.sub(' ', t.strip())
    t = digit_re.sub('', t)
    return t.strip()


def train_models(train_files, model_dir, N=3, cutoff=10000, token_mode='char', padding=False):
    if not path.exists(train_files):
        stderr.write('Train directory does not exist: {0}.\nUse the --train-files option to specify it.\n'.format(args.train_files))
        return
    stderr.write('Training models...\n')
    for lang in listdir(train_files):
        stderr.write(lang + '\n')
        with open(path.join(train_files, lang)) as f:
            text = f.read().decode('utf8')[:cutoff]
            ngrams = get_ngrams_from_text(text, token_mode, N, padding)
            model = get_probabilities(ngrams, N)
            write_model(model, lang, model_dir, token_mode)


def get_ngrams_from_text(text, token_mode, N, padding):
    sentences = get_tokens(clean_text(text), token_mode)
    if padding is True:
        if token_mode == 'char':
            pad = ' '
        else:
            pad = '*PADDING*'
    else:
        pad = False
    ngrams = get_seen_ngrams(sentences, N, pad)
    return ngrams


def get_tokens(text, token_mode, padding=True):
    if token_mode == 'char':
        return tokenize_char(text)
    elif token_mode == 'word':
        return tokenize_words(text)


def tokenize_char(text):
    return whitespace_re.split(text)


def tokenize_words(text):
    sentences = []
    for sen in sentence_re.finditer(text):
        sentences.append(whitespace_re.split(text))
    return sentences


def get_probabilities(ngrams, N, discount=0.5):
    probs = {}
    compute_unigram_probs(probs, ngrams)
    alphabet = set(probs.keys())
    for n in range(2, N + 1):
        compute_katz_probs(probs, ngrams, n, alphabet, discount)
    return probs


def compute_katz_probs(probs, ngrams, n, alphabet, discount=0.5):
    leftover = defaultdict(float)
    # iterate seen ngrams and compute probability
    for ngram, count in ngrams.iteritems():
        if len(ngram) != n:
            continue
        if ngram in probs:
            stderr.write('Something bad happened during training. Please report this message\n')
        pr = float(count - discount) / ngrams[ngram[:-1]]
        probs[ngram] = math.log(pr)
        leftover[ngram[:-1]] += pr
    missing = defaultdict(float)
    # probability of every other ngram
    for ngram_ in itertools.product(alphabet, repeat=n):
        ngram = tuple(t[0] for t in ngram_)
        if ngram in probs:
            continue
        missing[ngram[:-1]] += math.exp(probs[ngram[1:]])
    for ngram_ in itertools.product(alphabet, repeat=n):
        ngram = tuple(t[0] for t in ngram_)
        if ngram in probs:
            continue
        try:
            probs[ngram] = math.log((1 - leftover[ngram[:-1]])) + probs[ngram[1:]] - math.log(missing[ngram[:-1]])
        except:
            stderr.write(str(leftover[ngram[:-1]]) + ' ' + str(ngram) + '\n')


def compute_unigram_probs(probs, ngrams):
    unigram_sum = sum((v for k, v in ngrams.iteritems() if len(k) == 1))
    for ngram, count in ngrams.iteritems():
        # skip if not a unigram
        if len(ngram) > 1:
            continue
        probs[ngram] = math.log(float(count) / unigram_sum)


def get_seen_ngrams(text, N=3, padding=False):
    ngrams = defaultdict(int)
    for sentence in text:
        if padding:
            sen = [padding] * (N - 1) + list(sentence) + [padding] * (N - 1)
        else:
            sen = list(sentence)
        for n in xrange(1, N + 1):
            for i in xrange(0, len(sen) - n):
                ngrams[tuple(sen[i:i + n])] += 1
    return ngrams


def write_model(model, lang, model_dir, token_mode='char'):
    with open(path.join(args.model_dir, lang + '.model'), 'w') as f:
        if token_mode == 'char':
            f.write('\n'.join(u'{0}\t{1}'.format(''.join(ngram), prob) for ngram, prob in sorted(model.iteritems(), key=lambda x: -x[1])).encode('utf8') + '\n')
        elif token_mode == 'word':
            word_map = {}
            word_i = 0
            for ngram, prob in model.iteritems():
                for word in ngram:
                    if not word in word_map:
                        word_map[word] = word_i
                        word_i += 1
                f.write(u'{0}\t{1}\n'.format(' '.join(str(word_map[w]) for w in ngram), prob).encode('utf8'))
            #f.write('\n'.join(u'{0}\t{1}'.format(' '.join(ngram), prob) for ngram, prob in sorted(model.iteritems(), key=lambda x: -x[1])).encode('utf8') + '\n')
            with open(path.join(args.model_dir, lang + '.word_labels'), 'w') as g:
                g.write('\n'.join(u'{0}\t{1}'.format(v, k) for k, v in word_map.iteritems()).encode('utf8'))
        stderr.write('Model written to file: {0}\n'.format(path.join(model_dir, lang + '.model')))


def read_models(model_dir, filter_langs=set(), token_mode='char'):
    models = {}
    for fn in listdir(model_dir):
        if not 'model' in fn:
            continue
        lang = fn.rstrip('model')[:-1]
        models[lang] = {}
        if filter_langs and not lang in filter_langs:
            stderr.write('Skipping {0}\n'.format(lang))
            continue
        stderr.write('Reading {0}\n'.format(fn))
        with open(path.join(model_dir, fn)) as f:
            if token_mode == 'char':
                models[lang] = read_char_model(f)
            elif token_mode == 'word':
                map_fn = path.join(model_dir, lang) + '.word_labels'
                models[lang] = read_word_model(f, map_fn)

    return models


def read_word_model(stream, map_fn):
    model = {}
    with open(map_fn) as f:
        word_map = {}
        for l in f:
            fs = l.decode('utf8').strip().split('\t')
            word_map[fs[0]] = fs[1]
    for l in stream:
        ngram_, prob = l.decode('utf8').split('\t')
        ngram = tuple(word_map[n] for n in ngram_.split())
        model[ngram] = float(prob)
    return model


def read_char_model(stream):
    model = {}
    for l in stream:
        ngram, prob = l.decode('utf8').split('\t')
        ngram = tuple(ngram)
        model[ngram] = float(prob)
    return model


args = parse_args()


def main():
    if args.mode == 'train':
        train_models(args.train_files, args.model_dir, args.N, args.cutoff, args.token_mode, padding=True)
    else:
        lang_filt = set(args.filter_langs.split(','))
        models = read_models(args.model_dir, lang_filt, args.token_mode)
        identify_input(models)

if __name__ == '__main__':
    main()
