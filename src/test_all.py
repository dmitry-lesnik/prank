from text_generator import *
from helpers import *

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

        gen = Text_generator()
        eqs = Equivalent_sentences()

        template = '{This} function {increases} from $aa to $bb.'
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
        template = '{&Shut} that fucking {door} please.'
        s = gen.generate_sentence_from_template(template, self.vars)
        log_debug(s, 'generated sentence', std_out=True)
        assert s == 'Close that fucking door please.'

        s = gen.generate_sentence_from_template(template, self.vars)
        log_debug(s, 'generated sentence', std_out=True)
        assert s == 'Close that fucking door please.'

        s = gen.generate_sentence_from_template(template, self.vars)
        log_debug(s, 'generated sentence', std_out=True)
        assert s == 'Shut that fucking door please.'

        template = '$st {increases} from $aa to $bb.'
        s = gen.generate_sentence_from_template(template, self.vars)
        log_debug(s, 'generated sentence', std_out=True)
        assert s == 'This function grows from 8 to 12.'



        ####################################
        t = eqs.get_random_template('**value_is_growing|$st|$aa|$bb')
        log_debug(t, 'template', std_out=True)
        assert t == '{Next we observe that} $st {raises} in the interval from $aa to $bb.'
        s = gen.generate_sentence_from_template(t, self.vars)
        log_debug(s, '\tgenerated sentence', std_out=True)
        assert s == 'Next we observe that this function raises in the interval from 8 to 12.'


        t = eqs.get_random_template('**value_is_growing|$st|$aa|$bb')
        log_debug(t, 'template', std_out=True)
        assert t == '{Next we observe that} $st {raises} in the interval from $aa to $bb.'
        s = gen.generate_sentence_from_template(t, self.vars)
        log_debug(s, '\tgenerated sentence', std_out=True)
        assert s == 'Next we observe that this function grows in the interval from 8 to 12.'

        t = eqs.get_random_template('**value_is_growing|$st|$aa|$bb')
        log_debug(t, 'template', std_out=True)
        assert t == 'Notice that on the interval [$aa, $bb] $st is {growing}.'
        s = gen.generate_sentence_from_template(t, self.vars)
        log_debug(s, '\tgenerated sentence', std_out=True)
        assert s == 'Notice that on the interval [8, 12] this function is raising.'

        t = eqs.get_random_template('{&Shut} that fucking {door} please.')
        log_debug(t, 'template', std_out=True)
        assert t == '{&Shut} that fucking {door} please.'
        s = gen.generate_sentence_from_template(t, self.vars)
        log_debug(s, '\tgenerated sentence', std_out=True)
        assert s == 'Close that fucking door please.'


    def test_generate_block(self):

        random.seed(123)

        gen = Text_generator()

        proto_text = ['{This} function {increases} from $x to $y.',
                      '{&Shut} that fucking {door} please.',
                      '**value_is_growing|$w|$x|$y',
                      '',
                      'The value of $w {increases} from $x to $y.',
                      '{&Shut} that fucking {door} please.',
                      '**value_is_growing|$w|$x|$y']

        text = gen.generate_block(proto_text, self.vars)
        log_debug(text, 'text', std_out=True)

        expected_text = 'This function increases from 7 to 11. Shut that fucking door please. ' \
                        'Notice that on the interval [7, 11] $\omega$ is raising.\n\n' \
                        'The value of $\omega$ increases from 7 to 11. Shut that fucking door please. ' \
                        'Next we observe that $\omega$ increases in the interval from 7 to 11.'
        assert text == expected_text



if __name__ == '__main__':

    open_logger('test_all.log')

    T = TestGenerator()
    T.test_generate_sentence_from_template()
    T.test_generate_block()





