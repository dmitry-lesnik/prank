import numpy as np
from helpers import *
from string import Template

import random


CAPITAL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


class Synonyms(object):
    def __init__(self):
        self.synonyms = []
        self.synonyms.append(['buy', 'purchase'])
        self.synonyms.append(['big', 'large'])
        self.synonyms.append(['on', 'upon'])
        self.synonyms.append(['increases', 'raises'])
        self.synonyms.append(['increasing', 'raising'])
        self.synonyms.append(['decreases', 'falls off', 'decays'])
        self.synonyms.append(['decreasing', 'decaying'])


    def get_synonyms(self, w):
        for words in self.synonyms:
            for word in words:
                if w == word:
                    return words
        return [w]

    def get_random_synonym(self, w):
        decapitalise = lambda s: s[:1].lower() + s[1:] if s else ''
        capitalise = lambda s: s[:1].upper() + s[1:] if s else ''
        cap = False
        if w[0] in CAPITAL_LETTERS:
            w = decapitalise(w)
            cap = True



        words = self.get_synonyms(w)
        i = random.randint(0, len(words)-1)
        if cap:
            return capitalise(words[i])
        else:
            return words[i]



class test_generator(object):
    def __init__(self):
        pass


    def generate_sentence(self, words):
        """

        words: list[string]
            a list of words, some of which are capitalised,
            some words are punctuation symbols: commas and other, colons, etc

        Returns
        -------
        sentence : string
            words put together
        """
        pass


def parse_template(s):
    """
    template is a sentence with some words in curly brackets, like
    "This function is {raising} on the interval from a to b."

    Returns
    -------
    preformatted:   string
        sentence suitable for formatting, like
        "This function is {} on the interval from a to b."
    keys :  list
        keys to be used in formatting:
        preformatted.format(keys)
    """
    blocks = []
    keys = []
    for w in s.split('{'):
        w1 = w.split('}')
        if len(w1) == 1:
            blocks.append(w1[0])
        elif len(w1) == 2:
            if w1[0] == '':
                raise RuntimeError('zero length modifiable in curly brackets')

            keys.append(w1[0])
            blocks.append(w1[1])
    preformatted = '{}'.join(blocks)
    return preformatted, keys


def generate_sentence_from_preformatted(preformatted, keys):
    return preformatted.format(*tuple(keys))


def generate_sentence_from_template(template, vars=None, randomise=True):
    """
    Template is a sentence, which contains replaceable words in curly brackets
    "This function is {increasing} on the interval a to b"
    """

    preformatted, keys = parse_template(template)

    if randomise:
        syn = Synonyms()
        for idx, w in enumerate(keys):
            w = syn.get_random_synonym(w)
            keys[idx] = w
    s = generate_sentence_from_preformatted(preformatted, keys)
    if vars is not None:
        s = Template(s).safe_substitute(**vars)
    return s




if __name__ == "__main__":
    open_logger('text_generator.log')
    # random.seed(124)


    ###########################################

    a = 8
    b = 12
    vars = dict()
    vars['a'] = a
    vars['b'] = b

    template = '{This} function {increases} from $a to $b.'

    s = generate_sentence_from_template(template, vars)
    log_debug(s, 'generated sentence', std_out=True)








