from helpers import *
from text_generator import *
from post_processing import *




###################################################################



open_logger('generate_introduction.log')

random.seed(102)

vars = dict()
vars['aa'] = 8
vars['bb'] = 12
vars['st'] = 'this function'
vars['x'] = 7
vars['y'] = 11
vars['w'] = '$\omega$'


gen = Text_generator()
v = Variants()


v.read_from_file('introduction_variants2.txt')

log_debug(v.variants_list, "v.variants_list")

p = v.random_variant()
log_debug(p, 'random variant')
t = gen.generate_block(p, vars)
log_debug(t, 'generated text', std_out=True)



#################################################################
####### numerical results postprocessing ########################

log_debug('\n------------------------')
log_debug('post-processing\n')

x = np.linspace(0., 5., 21)
y = (x - 2.) ** 3 + 0.01 * x - 3. * np.exp(2. * (x - 4.))

n = x.size
n1 = 10 * n

grad_x, grad_y, grad_sign = filtered_gradient(x, y, n1)

limits_x, signs = find_regions(grad_x, grad_sign)
log_debug(limits_x, 'limits_x')
log_debug(signs, 'signs')

intervals = monotonous_intervals(limits_x, signs)

log_debug(intervals, 'intervals', std_out=True)
sig = get_signature(intervals)
log_debug(sig, 'signature', std_out=True)


###########################################################
########## simple narrative ###############################

p = []
for interval in intervals:
    sign = interval[0]
    if sign == 1:
        template = '**value_is_growing|$w|{:.3g}|{:.3g}'.format(interval[1], interval[2])
        p.append(template)
    if sign == -1:
        template = '**value_is_decreasing|$w|{:.3g}|{:.3g}'.format(interval[1], interval[2])
        p.append(template)
log_debug(p, 'post-processing output template', std_out=True)
t = gen.generate_block(p, vars)
log_debug(t, 'generated text', std_out=True)





