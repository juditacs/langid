import os
import math
from collections import defaultdict
from itertools import product


class Probability(object):

    def __init__(self, prob, missing=None):
        self.prob = prob
        self.missing = missing

    def __str__(self):
        if self.missing is None:
            return str(self.prob)
        return '{0}, missing: {1}'.format(math.exp(self.prob), self.missing)


class LanguageModel(object):

    def __init__(self, name, tokenizer=None):
        self.name = name
        self.tokenizer = tokenizer

    def train_model(self, stream, mode, N=3):
        tokens = self.tokenizer.tokenize(stream.read().decode('utf8'))
        self.N = N
        self.mode = mode
        ngrams = self.count_ngrams(tokens)
        self.compute_probs(ngrams)

    def count_ngrams(self, tokens):
        ngrams = defaultdict(int)
        for i in xrange(len(tokens)):
            for j in xrange(1, self.N + 1):
                if i + j > len(tokens):
                    continue
                ngram = tuple(tokens[i:i + j])
                ngrams[ngram] += 1
        return ngrams

    def compute_probs(self, ngrams):
        if self.mode == "normal":
            self.compute_simple_probs(ngrams)
        elif self.mode == "katz":
            self.compute_katz_probs(ngrams)

    def compute_simple_probs(self, ngrams):
        self.probs = {}
        unigram_sum = float(sum(v for k, v in ngrams.iteritems() if len(k) == 1))
        for k, v in ngrams.iteritems():
            if len(k) == 0:
                raise ValueError("Weird ngram: {}".format(k.encode('utf8')))
            if len(k) == 1:
                self.probs[k] = v / unigram_sum
            else:
                prefix_cnt = ngrams[k[:-1]]
                self.probs[k] = float(v) / prefix_cnt

    def compute_katz_probs(self, ngrams, d=0.5):
        self.probs = {}
        unigram_sum = float(sum(v for k, v in ngrams.iteritems() if len(k) == 1))
        alphabet = set()
        for k, v in ngrams.iteritems():
            if len(k) == 1:
                self.probs[k] = v / unigram_sum
                alphabet.add(k)
        for i in range(2, self.N + 1):
            missing = defaultdict(float)
            for ngram, cnt in ngrams.iteritems():
                if not len(ngram) == i:
                    continue
                pr = float(cnt - d) / ngrams[ngram[:-1]]
                missing[ngram[:-1]] += pr
                self.probs[ngram] = pr
            for ngr in product(alphabet, repeat=i):
                if ngr in self.probs:
                    continue
                ngr = tuple(n[0] for n in ngr)
                self.probs[ngr] = self.probs[ngr[1:]] * (1 - missing[ngr[:-1]])

    def write_model(self, stream, sort=False):
        if sorted:
            for ngr, prob in sorted(self.probs.iteritems(), key=lambda x: -x[1]):
                stream.write(u'{0}\t{1}\n'.format(''.join(ngr), prob).encode('utf8'))
        else:
            for ngr, prob in self.probs.iteritems():
                stream.write(u'{0}\t{1}\n'.format(''.join(ngr), prob).encode('utf8'))

    def eval_tokens(self, tokens):
        pr = 0.0
        missing = 0
        for i in xrange(len(tokens) - self.N + 1):
            ngr = tuple(tokens[i:i + self.N])
            if ngr in self.probs:
                pr += math.log(self.probs[ngr])
            else:
                missing += 1
        return Probability(pr, missing)


class Languages(object):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def train_models(self, basedir, mode, N=3):
        self.models = {}
        for fn in os.listdir(basedir):
            langname = Languages.get_langname_from_fn(fn)
            with open(os.path.join(basedir, fn)) as f:
                self.models[langname] = LanguageModel(langname, self.tokenizer)
                self.models[langname].train_model(f, mode, N)

    def write_models(self, basedir, prefix=""):
        for langname, model in self.models.iteritems():
            fn = os.path.join(basedir, prefix + langname)
            with open(fn, 'w') as f:
                model.write_model(f, sort=True)

    def classify_text(self, text):
        tokens = self.tokenizer.tokenize(text)
        pr = {}
        for lang, model in self.models.iteritems():
            pr[lang] = model.eval_tokens(tokens)
        return pr

    @staticmethod
    def get_langname_from_fn(fn):
        return fn.split('.')[-1]
