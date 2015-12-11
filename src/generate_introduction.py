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
s = Sections()
s.read_from_file()





#################################################################
####### numerical results postprocessing ########################

log_debug('\n------------------------')
log_debug('post-processing\n')

filename = 'output_1.txt'

A_full, A_num, Param  = read_calculation_output2(filename)

log_debug(A_full, 'A_full', std_out=True)
log_debug(A_num, 'A_num', std_out=True)


######################################
# analyse potential


pot_sign = A_full['potential']


# list of parameter values, where time-domain values differ more than 10% from WKB values

data = A_num.values
N = data.shape[0]

w_re = data[:, 3]
w_im = data[:, 5]
t_re = data[:, 4]
t_im = data[:, 6]
r1 = np.sqrt(w_re**2 + w_im**2)
r2 = np.sqrt(t_re**2 + t_im**2)
d = np.sqrt( (t_re - w_re)**2 + (t_im - w_im)**2)
rel_d = 2. * d / (r1 + r2 + 1e-12)
q_points = (rel_d > 0.001)






###########################################################
########## simple narrative ###############################

# p = []
# for interval in intervals:
#     sign = interval[0]
#     if sign == 1:
#         template = '**value_is_growing|$w|{:.3g}|{:.3g}'.format(interval[1], interval[2])
#         p.append(template)
#     if sign == -1:
#         template = '**value_is_decreasing|$w|{:.3g}|{:.3g}'.format(interval[1], interval[2])
#         p.append(template)
# log_debug(p, 'post-processing output template', std_out=True)
# t = gen.generate_block(p, vars)
# log_debug(t, 'generated text', std_out=True)





