import numpy as np
from helpers import *

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




if __name__ == "__main__":
    open_logger('text_generator.log')
    random.seed(1234)



    Syn = Synonyms()

    syn = Syn.get_synonyms('decreases')
    log_debug(syn, 'synonyms', std_out=True)

    syn = Syn.get_synonyms('decrease')
    log_debug(syn, 'synonyms', std_out=True)


    w = Syn.get_random_synonym('Decreases')
    log_debug(w, 'random synonym', std_out=True)

    w = Syn.get_random_synonym('DecreaseS')
    log_debug(w, 'random synonym', std_out=True)

    # use string.Template()














