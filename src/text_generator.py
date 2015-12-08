import numpy as np
from helpers import *
from string import Template
from read_methods import *


import random
import os, sys


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


class keywords_list(object):
    def __init__(self):
        self.var_start_key = '$begin_variants'
        self.var_end_key = '$end_variants'
        self.item_start_key = '${'
        self.item_end_key = '$}'
        self.section_begin_key = '$begin_section'
        self.section_end_key = '$end_section'


class Synonyms(object):
    def __init__(self, filename=None):

        if filename is None:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))
            filename = '/'.join([BASE_DIR, 'templates', 'synonyms.txt'])


        self.capital_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

        self.synonyms = []
        self.synonyms_ambiguous = dict()

        ret = read_synonyms(filename)
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
    def __init__(self, filename=None):

        if filename is None:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))
            filename = '/'.join([BASE_DIR, 'templates', 'sentence_templates.txt'])

        self.groups = read_templates(filename)

    def get_sentences_list(self, key):
        if key in self.groups:
            return self.groups[key]
        else:
            return [key]

    def get_random_template(self, metatemplate):
        w = metatemplate.strip('\n').split('|')
        sentences = self.get_sentences_list(w[0])
        i = 0
        if len(sentences) > 1:
            i = random.randint(0, len(sentences)-1)
        sentence_tmp = sentences[i]
        if metatemplate[-1] == '\n':
            sentence_tmp = ''.join([sentence_tmp, '\n'])
        if len(w) > 1:
            vars = dict()
            for j in range(1, len(w)):
                pn = 'p{}'.format(j)
                vars[pn] = w[j]
            sentence_tmp = Template(sentence_tmp).safe_substitute(**vars)
        return sentence_tmp


class Variants(object):
    """
    variant:            list
        a list of sentence templates or other variants
    variants_list :     list
        a list of variants

    """
    def __init__(self):
        self.variants_list = []

        self.kw = keywords_list()
        # self.var_start_key = '$begin_variants'
        # self.var_end_key = '$end_variants'
        # self.item_start_key = '${'
        # self.item_end_key = '$}'


    def read_from_file_2(self, filename=None):

        if filename is None:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))
            filename = '/'.join([BASE_DIR, 'templates', 'variants.txt'])

        f = open(filename)
        self.variants_list = []
        item = []

        while True:
            line = f.readline()
            line = line.strip(' \t')
            if line == '':
                break
            if line[0:len(self.kw.var_end_key)] == self.kw.var_end_key:
                raise RuntimeError('unexpected key {}'.format(self.kw.var_end_key))
            if line[0:len(self.kw.item_start_key)] == self.kw.item_start_key:
                raise RuntimeError('unexpected key {}'.format(self.kw.item_start_key))
            if line[0:len(self.kw.item_end_key)] == self.kw.item_end_key:
                raise RuntimeError('unexpected key {}'.format(self.kw.item_end_key))
            if line[0] == '#':
                continue
            if line[0:len(self.kw.var_start_key)] == self.kw.var_start_key:
                tmp = Variants()
                tmp.read_items_from_stream(f)
                item.append(tmp)
                continue

            item.append(line)

        self.variants_list.append(item)
        f.close()
        return self.variants_list


    def read_from_file(self, filename=None):

        if filename is None:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))
            filename = '/'.join([BASE_DIR, 'templates', 'variants.txt'])

        f = open(filename)
        self.variants_list = []

        while True:
            line = f.readline()
            if line == '':
                break
            line = line.strip(' \t\n')
            if line == '':
                continue
            if line[0:len(self.kw.var_end_key)] == self.kw.var_end_key:
                raise RuntimeError('unexpected key {}'.format(self.kw.var_end_key))
            if line[0:len(self.kw.item_start_key)] == self.kw.item_start_key:
                raise RuntimeError('unexpected key {}'.format(self.kw.item_start_key))
            if line[0:len(self.kw.item_end_key)] == self.kw.item_end_key:
                raise RuntimeError('unexpected key {}'.format(self.kw.item_end_key))
            if line[0] == '#':
                continue
            if line[0:len(self.kw.var_start_key)] == self.kw.var_start_key:
                self.read_items_from_stream(f)
                break
        f.close()
        return self.variants_list

    def read_items_from_stream(self, f):
        self.variants_list = []
        while True:
            line = f.readline()
            if line == '':
                raise RuntimeError('unexpected EOF. Missing key {}'.format(self.kw.var_end_key))
            line = line.strip(' \n\t')
            if line[0:len(self.kw.var_end_key)] == self.kw.var_end_key:
                break
            if line[0:len(self.kw.var_start_key)] == self.kw.var_start_key:
                raise RuntimeError('unexpected {}'.format(self.kw.var_start_key))
            if line[0:len(self.kw.item_start_key)] == self.kw.item_start_key:
                item = []
                while True:
                    line = f.readline()
                    line = line.strip(' \t')
                    if line == '':
                        raise RuntimeError('unexpected EOF. Missing key {}'.format(self.kw.item_end_key))

                    if line[0:len(self.kw.item_end_key)] == self.kw.item_end_key:
                        break
                    if line[0:len(self.kw.var_end_key)] == self.kw.var_end_key:
                        raise RuntimeError('unexpected key {}'.format(self.kw.var_end_key))
                    if line[0] == '#':
                        continue
                    if line[0:len(self.kw.var_start_key)] == self.kw.var_start_key:
                        tmp = Variants()
                        tmp.read_items_from_stream(f)
                        item.append(tmp)
                        continue
                    item.append(line)
                self.variants_list.append(item)
        return self.variants_list

    def random_variant(self):
        """
        return a block of text template
        """

        num = len(self.variants_list)
        i = 0
        if num > 1:
            i = random.randint(0, num-1)
        variant = self.variants_list[i]
        ret = []
        for line in variant:
            if isinstance(line, basestring):
                ret.append(line)
            elif isinstance(line, Variants):
                ret = ret + line.random_variant()
            else:
                raise RuntimeError('unexpected data type in variant list')
        return ret


class Sections(object):
    def __init__(self):
        self.sections = dict()
        self.kw = keywords_list()

    def read_from_file(self, filename=None, reset=True):
        if reset:
            self.sections = dict()

        if filename is None:
            BASE_DIR = os.path.dirname(os.path.dirname(__file__))
            filename = '/'.join([BASE_DIR, 'templates', 'sections_list.txt'])
        f = open(filename)

        while True:
            line = f.readline()
            line = line.strip(' \t')
            if line == '':
                break
            if line[0] == '#':
                continue

            if line[0:len(self.kw.section_begin_key)] == self.kw.section_begin_key:
                section = []
                key = line[len(self.kw.section_begin_key):].strip('}{\n')
                if key == '':
                    raise RuntimeError('missing section name')

                while True:
                    line = f.readline()
                    line = line.strip(' \t')
                    if line == '':
                        break
                    if line[0] == '#':
                        continue
                    if line[0:len(self.kw.section_end_key)] == self.kw.section_end_key:
                        break

                    if line[0:len(self.kw.var_start_key)] == self.kw.var_start_key:
                        tmp = Variants()
                        tmp.read_items_from_stream(f)
                        section.append(tmp)
                        continue
                    section.append(line)

                self.sections[key] = section

        return self.sections

    def get_section(self, section_name):
        if not section_name in self.sections:
            return []
        section = self.sections[section_name]

        ret = []
        for line in section:
            if isinstance(line, basestring):
                ret.append(line)
            elif isinstance(line, Variants):
                ret = ret + line.random_variant()
            else:
                raise RuntimeError('unexpected data type in section')
        return ret






class Text_generator(object):

    def __init__(self, synonyms_filename=None, eq_sent_filename=None):
        self.syn = Synonyms(synonyms_filename)
        self.eqs = Equivalent_sentences(eq_sent_filename)

    def parse_template(self, s):
        """
        template is a sentence with some words in curly brackets, like
        "This function is <<raising>> on the interval from a to b."

        Returns
        -------
        preformatted:   string
            sentence suitable for formatting, like
            "This function is {} on the interval from a to b."
        keys :  list
            keys to be used in formatting:
            preformatted.format(keys)
        """

        s = s.replace('{', '{{')
        s = s.replace('}', '}}')
        blocks = []
        keys = []
        for w in s.split('<<'):
            w1 = w.split('>>')
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
        if cap_first:
            s = capitalise(s)
        return s

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

    random.seed(100)

    vars = dict()
    vars['aa'] = 8
    vars['bb'] = 12
    vars['st'] = 'this function'
    vars['x'] = 7
    vars['y'] = 11
    vars['w'] = '$\omega$'


    gen = Text_generator()
    v = Variants()

    #
    # v.read_from_file('introduction_variants2.txt')
    #
    # p = v.random_variant()
    # log_debug(p, 'random variant')
    # t = gen.generate_block(p, vars)
    # log_debug(t, 'generated text', std_out=True)
    #


    s = Sections()
    s.read_from_file()


    p = s.get_section('introduction')
    log_debug(p, 'section')
    t = gen.generate_block(p, vars)
    log_debug(t, 'generated text', std_out=True)


    p = s.get_section('conclusion1')
    log_debug(p, 'section')
    t = gen.generate_block(p, vars)
    log_debug(t, 'generated text', std_out=True)


