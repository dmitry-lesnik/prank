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

filename = 'calculation_output.txt'

ret = read_calculation_output(filename)
A_full = ret['A_full']
A_num = ret['A_num']
a = ret['param_a']
b = ret['param_b']
v_full = ret['values_full']
v_a0 = ret['values_slice_a0']
v_b0 = ret['values_slice_b0']

log_debug(A_full, 'A_full', std_out=True)
log_debug(A_num, 'A_num', std_out=True)


###############################
#### fixed a ##################

WKB_Re_a0 = v_a0[:, 0]
WKB_Im_a0 = v_a0[:, 1]
WKB_Re_b0 = v_b0[:, 0]
WKB_Im_b0 = v_b0[:, 1]


log_debug('\n============================================')
log_debug('=======  WKB: Re(omega)  ===================')

#--------------
log_debug('fixed a:')
intervals, signature = get_full_analysis(b, WKB_Re_a0)
log_debug(intervals, 'intervals')
log_debug(signature, 'signature')


#--------------
log_debug('fixed b:')
intervals, signature = get_full_analysis(a, WKB_Re_b0)
log_debug(intervals, 'intervals')
log_debug(signature, 'signature')



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





