from sys import argv, stderr
import readline
import math
from collections import defaultdict
from scipy.stats import rv_discrete


class LanguageModel(object):

    def __init__(self):
        self.probs = defaultdict(dict)
        self.ngram_map = defaultdict(dict)
        self.rvs = {}

    def create_rvs(self, ngram):
        pr = self.probs[ngram]
        x = []
        y = []
        for i, (k, v) in enumerate(pr.iteritems()):
            self.ngram_map[ngram][i] = k
            x.append(i)
            y.append(v)
        self.rvs[ngram] = rv_discrete(values=(x, y))

    def get_rvs(self, ngram):
        if not ngram in self.rvs:
            self.create_rvs(ngram)
        return self.rvs[ngram].rvs()

    def generate_next(self, ngram):
        r = self.get_rvs(ngram)
        return self.ngram_map[ngram][r]


def read_model(fn):
    lm = LanguageModel()
    with open(fn) as f:
        for l in f:
            try:
                fd = l.decode('utf8').split('\t')
                ngram = fd[0]
                #if not ngram.strip() or not ngram[:-1].strip():
                    #continue
                pr = math.exp(float(fd[1]))
                lm.probs[ngram[:-1]][ngram[-1]] = pr
            except Exception as e:
                stderr.write('{0} {1} at line {2}\n'.format(type(e), e, l.strip()))
    return lm


def main():
    model = read_model(argv[1])
    N = int(argv[2]) if len(argv) > 2 else 3
    stderr.write('Model loaded\n')
    while(True):
        try:
            begin, l = raw_input('> ').decode('utf8').split()
            l = int(l)
            new_text = begin[:]
            ng = begin[-N + 1:]
            for i in xrange(l):
                next_char = model.generate_next(ng)
                ng = ng[1:] + next_char
                new_text += next_char
            print(new_text.encode('utf8'))
        except ValueError:
            stderr.write('Invalid input\n')

if __name__ == '__main__':
    main()
