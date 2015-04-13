import string
import re


class Tokenizer(object):

    """ tokenize text to characters or words
    Trim punctuation/whitespace/etc. if specified """

    def __init__(self, token, lower=False, filter_punct=False, filter_non_latin=False, ws_norm=False, strip=False):
        self.mode = token
        assert self.mode in ('char', 'word')
        # lower text
        self.to_lower = lower
        # replace all punctuation (string.punctuation) with a single space
        self.filter_punct = filter_punct
        # replace characters out of range 0x0000-0x24F range (latin extended) with a single space
        self.filter_non_latin = filter_non_latin
        # replace all whitespace with a single space
        # this always comes last
        self.ws_norm = ws_norm
        # trip leading and ending whitespace
        self.to_strip = strip
        self.setup_trim()

    def setup_trim(self):
        self.trim_chain = []
        if self.to_lower:
            self.trim_chain.append(lambda x: x.lower())
        if self.filter_punct:
            punct_re = re.compile(r'[{0}]'.format(re.escape(string.punctuation)), re.UNICODE)
            self.trim_chain.append(lambda x: punct_re.sub(' ', x))
        if self.ws_norm:
            ws_re = re.compile(r'\s+', re.UNICODE)
            self.trim_chain.append(lambda x: ws_re.sub(' ', x))
        if self.filter_non_latin:
            self.trim_chain.append(lambda x: ''.join(filter(lambda ch: ord(ch) <= 0x024F, x)))
        if self.to_strip:
            self.trim_chain.append(lambda x: x.strip())

    def trim_text(self, text):
        for method in self.trim_chain:
            text = method(text)
        return text

    def tokenize(self, text):
        trimmed = self.trim_text(text)
        if self.mode == 'char':
            return list(trimmed)
        elif self.mode == 'word':
            return trimmed.split()
