#!/usr/bin/env python2.7
from sys import exit, stderr
from os import listdir, path, makedirs
from argparse import ArgumentParser
import signal
import readline
from collections import defaultdict
from itertools import product
import re

def get_seen_ngrams(tokens_, n, cut=0):
    ngrams = defaultdict(int)
    if cut > 0:
        tokens = tokens_[0:cut+1]
    else:
        tokens = tokens_
    for k in range(1, n+1):
        for i, ngram in enumerate(tokens[k-1:]):
            ngrams[tokens[i:i+k]] += 1
    return ngrams

def signal_handler(signal, frame):
    print 'Kilepes...'
    exit(0)

def decide_lang(text, models, cfg):
    n = cfg.N
    probs = defaultdict(float)
    ngrams = get_seen_ngrams(text, n)
    if len(text) < n:
        for lang in models:
            probs[lang] = 1.0
            for ngram in ngrams:
                probs[lang] *= models[lang][ngram]
    else:
        for lang in models:
            probs[lang] = 1.0
            for ngram in ngrams:
                if len(ngram) != n:
                    continue
                if ngram in models[lang]:
                    probs[lang] *= models[lang][ngram]
                else:
                    if cfg.log > 1:
                        print lang.encode('utf8') + ": ngram nem szerepel a modellben: " + ngram.encode('utf8')
                    probs[lang] = 0.0
                    break
    max_prob = 0.0
    max_lang = ""
    for l in probs.keys():
        if probs[l] > max_prob:
            max_lang = l
            max_prob = probs[l]
    if cfg.log > 0:
        for l, pr in sorted(probs.iteritems(), key=lambda x: x[1]):
            print '{0} {1}'.format(l.encode('utf8'), pr)
    if max_prob == 0.0:
        print "P=0 minden nyelvre"
    else:
        print max_lang.lower()

def trim_text(text, cfg):
    text_ = text.decode('utf8')
    text_ = text_.replace(',', ' , ')
    text_ = text_.replace('\.', ' \. ')
    text_ = re.sub('[\s\n]+', ' ', text_, flags=re.UNICODE)
    if cfg.lower:
        text_ = text_.lower()
    if cfg.nospec:
        text_ = re.sub('[^\w\s\.\,]+', '', text_, flags=re.UNICODE)
        text_ = re.sub('[\d]+', '', text_, flags=re.UNICODE)
    if cfg.nospace:
        text_ = re.sub('\s+', '', text_, flags=re.UNICODE)
    return text_

def get_prob(ngrams, prob_mode, alphabet, n=3):
    if prob_mode == "normal":
        return get_prob_simple(ngrams, n)
    elif prob_mode == "katz":
        return katz_backoff(ngrams, alphabet, n)

def get_prob_simple(ngrams, n=3):
    prob = defaultdict(float)
    ngram_sum = dict()
    for i in range(1, n+1):
        ngram_sum[i] = sum([v for (k, v) in ngrams.items() if len(k) == i])
    for ngram, cnt in ngrams.items():
        prob[ngram] = float(cnt)/ngram_sum[len(ngram)]
    return prob

def katz_backoff(ngrams, alphabet, n, discount=0.5):
    prob = defaultdict(float)
    # n=1
    unigram_sum = sum([v for k, v in ngrams.items() if len(k) == 1])
    for unigram, cnt in ngrams.items():
        if len(unigram) > 1:
            continue
        prob[unigram] = float(cnt)/unigram_sum
    # n>1
    for i in range(2, n+1):
        missing = defaultdict(float)
        for ngram, cnt in ngrams.items():
            if len(ngram) != i:
                continue
            prob[ngram] = float(cnt-discount)/ngrams[ngram[0:-1]]
            missing[ngram[0:-1]] += prob[ngram]
        for ngr_ in product(alphabet, repeat=i):
            ngr = ''.join(ngr_)
            if ngr in ngrams:
                continue
            prob[ngr] = (1 - missing[ngr[0:-1]]) * prob[ngr[1:]]
    return prob

def linear_interpolation(text, ngrams, weights):
    if len(weights) < 3:
        print "weights not specified"
        return 
    prob = 1.0
    unigram_sum = float(sum([v for k, v in ngrams.items() if len(k) == 1]))
    bigram_sum = float(sum([v for k, v in ngrams.items() if len(k) == 2]))
    trigram_sum = float(sum([v for k, v in ngrams.items() if len(k) == 3]))
    for k in range(0, len(text)-1):
        prob *= (weights[0] * ngrams[text[k:k+3]] / trigram_sum + \
                weights[1] * ngrams[text[k+1:k+3]] / bigram_sum + \
                weights[2] * ngrams[text[k+2:k+3]] / unigram_sum)
    return prob

def setup_parser():
    parser = ArgumentParser()
    parser.add_argument('-t', '--train', dest='train', default='train_files')
    parser.add_argument('-s', '--nospecial', dest='nospec', action='store_true')
    parser.add_argument('--lower', dest='lower', action='store_true', default=False)
    parser.add_argument('--noenter', dest='noenter', action='store_true', default=False)
    parser.add_argument('--nospace', dest='nospace', action='store_true', default=False)
    parser.add_argument('-w', '--write-models', dest='write_models', type=str,
                       default='models/default', 
                        help='write trained models. Destination should be the prefix of files')
    parser.add_argument('-r', '--read-models', dest='read_models', type=str,
                       default='')
    parser.add_argument('--just-train', dest='just_train', action='store_true',
                        default=False)
    parser.add_argument('-l', '--log', dest='log', type=int, default=0, 
                       help='logging level')
    parser.add_argument('-c', '--cutoff', dest='cutoff', default=0, type=int)
    parser.add_argument('-N', '--N', dest='N', default=3, type=int)
    parser.add_argument('-k', '--k', dest='k', default=20, type=int)
    parser.add_argument('-m', '--mode', dest='mode', default='normal', choices=('normal', 'katz'))
    return parser

def read_probs(fn, probs):
    try:
        f = open(fn)
    except OSError:
        stderr.write('File not found {0}\n'.format(fn))

    for l in f:
        try:
            ngram, prob = l.decode('utf8').strip().split('\t')
            probs[ngram] = float(prob)
        except UnicodeDecodeError:
            stderr.write('UnicodeDecodeError on line: {0}'.format(l))
        except ValueError:
            stderr.write('ValueError on line: {0}'.format(l))
    try:
        f.close()
    except:
        pass

def get_k_langs(cfg):
    i = 0
    f = open(cfg.train)
    train_files = list()
    for l in f:
        i += 1
        if i > cfg.k:
            f.close()
            return train_files
        lang, fn = l.decode('utf8').strip().split('\t')
        train_files.append((lang, fn))
    f.close()
    return train_files

def main():
    parser = setup_parser()
    cfg = parser.parse_args()
    train_files = get_k_langs(cfg)
    probs = defaultdict(lambda: defaultdict(float))
    signal.signal(signal.SIGINT, signal_handler)
    ngrams = dict()

    if cfg.read_models:
        basedir = '/'.join(cfg.read_models.strip().split('/')[:-1])
        basedir = basedir if basedir else '.'
        file_base = cfg.read_models.split('/')[-1] + '.'
        for fn in listdir(basedir):
            if not fn.startswith(file_base):
                continue
            lang = fn.split(file_base)[-1].strip()
            read_probs(basedir + '/' + fn, probs[lang])
    else:
        for lang, fn in train_files:
            f = open(fn)
            print lang.encode('utf8')
            tokens = trim_text(f.read(), cfg)
            ngrams[lang] = get_seen_ngrams(tokens, cfg.N, cfg.cutoff)
            probs[lang] = get_prob(ngrams[lang], cfg.mode, set(tokens), cfg.N)
            f.close()

    if cfg.write_models:
        if not path.exists('/'.join(cfg.write_models.split('/')[:-1])):
            makedirs('/'.join(cfg.write_models.split('/')[:-1]))
        for lang in probs.keys():
            f = open(cfg.write_models + '.' + lang, 'w')
            f.write('\n'.join(['{0}\t{1}'.format(k.encode('utf8'), v) for k, v in probs[lang].items()]))
            f.close()
            print lang.encode('utf8') + u' modell elmentve: '.encode('utf8') + cfg.write_models + '.' + lang.encode('utf8')
        print "Modellek fajlba irva"
    print "Tanitas kesz."
        
    if cfg.just_train:
        exit(0)

    while True:
        try:
            unk = raw_input('> ')
            decide_lang(trim_text(unk, cfg), probs, cfg)
        except Exception:
            continue
    

if __name__ == '__main__':
    main()
