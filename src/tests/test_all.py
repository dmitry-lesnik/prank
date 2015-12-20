from src.text_generator import *
from src.helpers import *

class TestGenerator(object):
    def __init__(self):
        self.vars = dict()
        self.vars['aa'] = 8
        self.vars['bb'] = 12
        self.vars['st'] = 'this function'
        self.vars['x'] = 7
        self.vars['y'] = 11
        self.vars['w'] = '$\omega$'

    def test_generate_sentence_from_template(self):

        random.seed(119)

        gen = Text_generator(synonyms_filename='test_synonyms.txt')
        eqs = Equivalent_sentences()

        template = '<<This>> function <<increases>> from $aa to $bb.'
        s = gen.generate_sentence_from_template(template, self.vars)
        log_debug(s, 'generated sentence', std_out=True)
        assert s == 'This function increases from 8 to 12.'

        s = gen.generate_sentence_from_template(template, self.vars)
        log_debug(s, 'generated sentence', std_out=True)
        assert s == 'This function raises from 8 to 12.'

        s = gen.generate_sentence_from_template(template, self.vars)
        log_debug(s, 'generated sentence', std_out=True)
        assert s == 'This function raises from 8 to 12.'

        # test homonyms
        template = '<<&Shut>> that fucking <<door>> please.'
        s = gen.generate_sentence_from_template(template, self.vars)
        log_debug(s, 'generated sentence', std_out=True)
        assert s == 'Close that fucking door please.'

        s = gen.generate_sentence_from_template(template, self.vars)
        log_debug(s, 'generated sentence', std_out=True)
        assert s == 'Close that fucking door please.'

        s = gen.generate_sentence_from_template(template, self.vars)
        log_debug(s, 'generated sentence', std_out=True)
        assert s == 'Shut that fucking door please.'

        template = '$st <<increases>> from $aa to $bb.'
        s = gen.generate_sentence_from_template(template, self.vars)
        log_debug(s, 'generated sentence', std_out=True)
        assert s == 'This function grows from 8 to 12.'



        ####################################
        t = eqs.get_random_template('**value_is_growing|$st|$aa|$bb')
        log_debug(t, 'template', std_out=True)
        assert t == '<<Next we observe that>> $st <<raises>> in the interval from $aa to $bb.'
        s = gen.generate_sentence_from_template(t, self.vars)
        log_debug(s, '\tgenerated sentence', std_out=True)
        assert s == 'We have observed that this function raises in the interval from 8 to 12.'


        t = eqs.get_random_template('**value_is_growing|$st|$aa|$bb')
        log_debug(t, 'template', std_out=True)
        assert t == '<<Next we observe that>> $st <<raises>> in the interval from $aa to $bb.'
        s = gen.generate_sentence_from_template(t, self.vars)
        log_debug(s, '\tgenerated sentence', std_out=True)
        assert s == 'We have observed that this function grows in the interval from 8 to 12.'

        t = eqs.get_random_template('**value_is_growing|$st|$aa|$bb')
        log_debug(t, 'template', std_out=True)
        assert t == 'Notice that on the interval [$aa, $bb] $st is <<growing>>.'
        s = gen.generate_sentence_from_template(t, self.vars)
        log_debug(s, '\tgenerated sentence', std_out=True)
        assert s == 'Notice that on the interval [8, 12] this function is raising.'

        t = eqs.get_random_template('<<&Shut>> that fucking <<door>> please.')
        log_debug(t, 'template', std_out=True)
        assert t == '<<&Shut>> that fucking <<door>> please.'
        s = gen.generate_sentence_from_template(t, self.vars)
        log_debug(s, '\tgenerated sentence', std_out=True)
        assert s == 'Close that fucking door please.'


    def test_generate_block(self):

        random.seed(123)

        gen = Text_generator(synonyms_filename='test_synonyms.txt')

        proto_text = ['<<This>> function <<increases>> from $x to $y.',
                      '<<&Shut>> that fucking <<door>> please.',
                      '**value_is_growing|$w|$x|$y',
                      '',
                      'The value of $w <<increases>> from $x to $y.',
                      '<<&Shut>> that fucking <<door>> please.',
                      '**value_is_growing|$w|$x|$y']

        text = gen.generate_block(proto_text, self.vars)
        log_debug(text, 'text', std_out=True)

        expected_text = 'This function increases from 7 to 11. Close that fucking door please. ' \
                        'Notice that on the interval [7, 11] $\omega$ is increasing.\n\n' \
                        'The value of $\omega$ raises from 7 to 11. ' \
                        'Close that fucking door please. ' \
                        'We have observed that $\omega$ raises in the interval from 7 to 11.'
        assert text == expected_text


    def test_variants(self):


        random.seed(100)

        gen = Text_generator(synonyms_filename='test_synonyms.txt')

        v = Variants()
        ret = v.read_from_file('test_variants.txt')
        log_debug(ret, 'variants')

        p = v.random_variant()
        t = gen.generate_block(p, self.vars)
        log_debug(t, 'random text', std_out=True)
        expected_text = 'Variant 1 sentence 1.\nVariant 1 sentence 2.\nVariant 1 sentence 3.\n'
        assert t == expected_text


        p = v.random_variant()
        t = gen.generate_block(p, self.vars)
        log_debug(t, 'random text', std_out=True)
        expected_text = 'Variant 2 sentence 1.\nVariant 2 sentence 2.\n\nVariant 2 sentence 3.\n'
        assert t == expected_text

        p = v.random_variant()
        t = gen.generate_block(p, self.vars)
        log_debug(t, 'random text', std_out=True)
        expected_text = 'Variant 3 sentence 1.\nVariant 3.2 sentence 2.\n' \
                        'Notice that on the interval [7, 11] $\omega$ is raising.\n' \
                        'Variant 3 sentence 3.\n'
        assert t == expected_text

        p = v.random_variant()
        t = gen.generate_block(p, self.vars)
        log_debug(t, 'random text', std_out=True)
        expected_text = 'Variant 3 sentence 1.\nVariant 3.2 sentence 2.\n' \
                        'We have observed that $\omega$ increases in the interval from 7 to 11.\n' \
                        'Variant 3 sentence 3.\n'
        assert t == expected_text

    def test_sections(self):
        random.seed(101)

        gen = Text_generator(synonyms_filename='test_synonyms.txt')

        s = Sections()
        s.read_from_file('test_sections_list.txt')
        s.read_from_file('test_sections_list2.txt', reset=False)


        p = s.get_section('conclusion2')
        log_debug(p, 'section')
        t = gen.generate_block(p, self.vars)
        log_debug(t, 'generated text', std_out=True)

        expected_text = 'Most US Muslims have arrived in the US in recent decades, ' \
        'making them a relatively fresh immigrant group. ' \
        'Many say they are watching the billionaire real estate mogul-turned-reality ' \
        'TV star-turned rising political star with alarm.\n' \
        'AAAAA 2.\nAAAAA 2.\n'

        assert t == expected_text



if __name__ == '__main__':

    import os, sys
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    print BASE_DIR

    open_logger('test_all.log')

    T = TestGenerator()
    T.test_generate_sentence_from_template()
    T.test_generate_block()
    T.test_variants()
    T.test_sections()





