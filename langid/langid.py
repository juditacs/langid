from argparse import ArgumentParser
from sys import stdin

from tokenize import Tokenizer
from model import Languages


def parse_args():
    p = ArgumentParser()
    p.add_argument('--token', choices=['char', 'word'], default='char', help='Tokenize by word or character')
    p.add_argument('-l', '--lower', action='store_true', default=False, help='Lower all input')
    p.add_argument('--filter-punct', action='store_true', default=False, help='Replace punctuation with space')
    p.add_argument('--filter-non-latin', action='store_true', default=False, help='Replace non-Latin characters with space')
    p.add_argument('--normalize-whitespace', dest='ws_norm', action='store_true', default=False, help='Normalize whitespace')
    p.add_argument('--strip', action='store_true', default=False, help='Strip leading and trailing whitespace')
    p.add_argument('--train-dir', type=str, default='models', help='Location of training files')
    p.add_argument('--train-mode', choices=['normal', 'katz'], default='normal')
    return p.parse_args()


def main():
    cfg = parse_args()
    tokenizer = Tokenizer(cfg.token, cfg.lower, cfg.filter_punct, cfg.filter_non_latin, cfg.ws_norm, cfg.strip)
    lm = Languages(tokenizer)
    lm.train_models(cfg.train_dir, cfg.train_mode)
    #lm.write_models('models')
    for l in stdin:
        pr = lm.classify_text(l.decode('utf8'))
        print('\n'.join('{0}: {1}'.format(k, v) for k, v in pr.iteritems()))

if __name__ == '__main__':
    main()
