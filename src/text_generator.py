import numpy as np
from helpers import *
from string import Template
from read_methods import *


import random


# CAPITAL_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

def decapitalise(s, first_n=1):
    if s:
        return s[:first_n].lower() + s[first_n:]
    else:
        return ''

def capitalise(s):
    if s:
        return s[:1].upper() + s[1:]
    else:
        return ''




class Synonyms(object):
    def __init__(self):

        self.capital_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        self.synonyms = []
        self.synonyms_ambiguous = dict()

        ret = read_synonyms('synonyms.txt')
        for line in ret:
            if len(line) > 0:
                if line[0][0] == '&':
                    self.synonyms_ambiguous[line[0]] = line[1:]
                else:
                    self.synonyms.append(line)

    def get_synonyms_list(self, w):

        # first check homonyms
        if w[0] == '&':
            if w in self.synonyms_ambiguous:
                return self.synonyms_ambiguous[w]

        # now check other synonyms
        for words in self.synonyms:
            for word in words:
                if w == word:
                    return words
        return [w]

    def get_random_synonym(self, w):

        cap = False
        if w[0] in self.capital_letters:
            w = decapitalise(w)
            cap = True
        if w[0] == '&' and w[1] in self.capital_letters:
            w = decapitalise(w, first_n=2)
            cap = True

        words = self.get_synonyms_list(w)
        i = random.randint(0, len(words)-1)
        if cap:
            return capitalise(words[i])
        else:
            return words[i]


class Equivalent_sentences(object):
    def __init__(self):
        self.groups = read_templates('sentence_templates.txt')

    def get_sentences_list(self, key):
        if key in self.groups:
            return self.groups[key]
        else:
            return [key]

    def get_random_template(self, metatemplate):
        w = metatemplate.split('|')
        sentences = self.get_sentences_list(w[0])
        i = random.randint(0, len(sentences)-1)
        sentence_tmp = sentences[i]
        if len(w) > 1:
            vars = dict()
            for j in range(1, len(w)):
                pn = 'p{}'.format(j)
                vars[pn] = w[j]
            sentence_tmp = Template(sentence_tmp).safe_substitute(**vars)
        return sentence_tmp


class Variants(object):
    def __init__(self):
        self.block_template = []


    def read_from_file(self, filename):
        f = open(filename)


        f.close()




class Text_generator(object):
    def __init__(self):
        self.syn = Synonyms()
        self.eqs = Equivalent_sentences()



    def parse_template(self, s):
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

    def generate_sentence_from_preformatted(self, preformatted, keys):
        return preformatted.format(*tuple(keys))


    def generate_sentence_from_template(self, template, vars=None, cap_first=True):
        """
        Template is a sentence, which contains replaceable words in curly brackets
        "This function is {increasing} on the interval a to b"
        """
        preformatted, keys = self.parse_template(template)

        for idx, w in enumerate(keys):
            w = self.syn.get_random_synonym(w)
            keys[idx] = w
        s = self.generate_sentence_from_preformatted(preformatted, keys)
        if vars is not None:
            s = Template(s).safe_substitute(**vars)
        return capitalise(s)

    def generate_sentence_from_metatemplate(self, metatemplate, vars=None, cap_first=True):
        tmp = self.eqs.get_random_template(metatemplate)
        return self.generate_sentence_from_template(tmp, vars, cap_first)



    def generate_block(self, p, vars):
        """
        Params
        ------
        p:      list(string)
            Each string is a sentence, or sentence template
            Empty string indicates a new paragraph

        Returns
        -------
        ret:    string
            output text
        """

        output = ''
        for tmp in p:
            if tmp == '':
                output += '\n\n'
            else:
                t = self.eqs.get_random_template(tmp)
                s = self.generate_sentence_from_metatemplate(t, vars)
                if output == '':
                    output = s
                elif output[-1] == '\n':
                    output = "".join([output, s])
                else:
                    output = " ".join([output, s])

        return output


if __name__ == "__main__":
    open_logger('text_generator.log')


