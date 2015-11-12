import numpy as np
from helpers import *
from string import Template

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
        self.synonyms.append(['buy', 'purchase'])
        self.synonyms.append(['big', 'large'])
        self.synonyms.append(['on', 'upon'])
        self.synonyms.append(['increases', 'raises', 'grows'])
        self.synonyms.append(['increasing', 'raising', 'growing'])
        self.synonyms.append(['decreases', 'falls off', 'decays'])
        self.synonyms.append(['decreasing', 'decaying'])

        self.synonyms_ambiguous = dict()
        self.synonyms_ambiguous['$shut'] = ['close', 'shut']
        self.synonyms_ambiguous['$near'] = ['close', 'near']
        self.synonyms_ambiguous['$near'] = ['close', 'near']


    def get_synonyms_list(self, w):

        # first check homonyms
        if w[0] == '$':
            if w in self.synonyms_ambiguous:
                return self.synonyms_ambiguous[w]

        # now check other synonyms
        for words in self.synonyms:
            for word in words:
                if w == word:
                    return words
        return [w]

    def get_random_synonym(self, w):
        # decapitalise = lambda s: s[:1].lower() + s[1:] if s else ''
        # capitalise = lambda s: s[:1].upper() + s[1:] if s else ''

        cap = False
        if w[0] in self.capital_letters:
            w = decapitalise(w)
            cap = True
        if w[0] == '$' and w[1] in self.capital_letters:
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
        self.groups = dict()

        self.groups['**value_is_growing'] = ['{Next we observe that} $p1 {raises} in the interval from $p2 to $p3.',
                                              'Notice that on the interval [$p2, $p3] $p1 is {growing}.']

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



def test_narrative():
    a = 8
    b = 12
    vars = dict()
    vars['x'] = a
    vars['y'] = b
    vars['w'] = 'omega'

    gen = Text_generator()


    proto_text = ['{This} function {increases} from $x to $y.',
                  '{$Shut} that fucking {door} please.',
                  '**value_is_growing|$w|$x|$y',
                  'The value of $w {increases} from $x to $y.',
                  '{$Shut} that fucking {door} please.',
                  '**value_is_growing|$w|$x|$y']

    output = ''
    for tmp in proto_text:
        t = eqs.get_random_template(tmp)
        s = gen.generate_sentence_from_metatemplate(t, vars)
        if output == '':
            output = s
        else:
            output = " ".join([output, s])
        log_debug(output, 'output so far')



if __name__ == "__main__":
    open_logger('text_generator.log')
    random.seed(119)

    ###########################################

    a = 8
    b = 12
    vars = dict()
    vars['aa'] = a
    vars['bb'] = b
    vars['st'] = 'this function'

    gen = Text_generator()
    eqs = Equivalent_sentences()

    template = '{This} function {increases} from $aa to $bb.'
    s = gen.generate_sentence_from_template(template, vars)
    log_debug(s, 'generated sentence', std_out=True)
    assert s == 'This function increases from 8 to 12.'

    s = gen.generate_sentence_from_template(template, vars)
    log_debug(s, 'generated sentence', std_out=True)
    assert s == 'This function raises from 8 to 12.'

    s = gen.generate_sentence_from_template(template, vars)
    log_debug(s, 'generated sentence', std_out=True)
    assert s == 'This function raises from 8 to 12.'

    # test homonyms
    template = '{$Shut} that fucking {door} please.'
    s = gen.generate_sentence_from_template(template, vars)
    log_debug(s, 'generated sentence', std_out=True)
    assert s == 'Close that fucking door please.'

    s = gen.generate_sentence_from_template(template, vars)
    log_debug(s, 'generated sentence', std_out=True)
    assert s == 'Close that fucking door please.'

    s = gen.generate_sentence_from_template(template, vars)
    log_debug(s, 'generated sentence', std_out=True)
    assert s == 'Shut that fucking door please.'

    template = '$st {increases} from $aa to $bb.'
    s = gen.generate_sentence_from_template(template, vars)
    log_debug(s, 'generated sentence', std_out=True)
    assert s == 'This function grows from 8 to 12.'



    ####################################
    t = eqs.get_random_template('**value_is_growing|$st|$aa|$bb')
    log_debug(t, 'template', std_out=True)
    assert t == '{Next we observe that} $st {raises} in the interval from $aa to $bb.'
    s = gen.generate_sentence_from_template(t, vars)
    log_debug(s, '\tgenerated sentence', std_out=True)
    assert s == 'Next we observe that this function raises in the interval from 8 to 12.'


    t = eqs.get_random_template('**value_is_growing|$st|$aa|$bb')
    log_debug(t, 'template', std_out=True)
    assert t == '{Next we observe that} $st {raises} in the interval from $aa to $bb.'
    s = gen.generate_sentence_from_template(t, vars)
    log_debug(s, '\tgenerated sentence', std_out=True)
    assert s == 'Next we observe that this function grows in the interval from 8 to 12.'

    t = eqs.get_random_template('**value_is_growing|$st|$aa|$bb')
    log_debug(t, 'template', std_out=True)
    assert t == 'Notice that on the interval [$aa, $bb] $st is {growing}.'
    s = gen.generate_sentence_from_template(t, vars)
    log_debug(s, '\tgenerated sentence', std_out=True)
    assert s == 'Notice that on the interval [8, 12] this function is raising.'

    t = eqs.get_random_template('{$Shut} that fucking {door} please.')
    log_debug(t, 'template', std_out=True)
    assert t == '{$Shut} that fucking {door} please.'
    s = gen.generate_sentence_from_template(t, vars)
    log_debug(s, '\tgenerated sentence', std_out=True)
    assert s == 'Close that fucking door please.'



    test_narrative()













