from helpers import *
from text_generator import *
from post_processing import *


def find_true_interval(x, Param=None):
    """
    x - Series:
        index:  array(float)[N]
        values: array(Boolean)[N]

    Returns
    -------
    ret:    list
        ret[0]:     string {'connected', 'disconnected', 'all-true', 'all-false'}
        'true' if all x == true
        'false' if all x == false
        'connected' if interval where x==True is 1-connected
        if 'connected' then
        [ret[1], ret[2]] - interval of indices where x == True
        'disconnected' if interval where x==True is not 1-connected
    """
    if all(x):
        return ['all-true']
    if all(-x):
        return ['all-false']

    n = x.size
    if Param is None:
        Param = range(n)

    ret = []
    i = 0
    while i<n and not x.iloc[i]:
        i += 1

    p1_idx = i

    j = i
    while j<n and x.iloc[j]:
        j += 1

    p2_idx = j-1
    while j<n:
        if x.iloc[j]:
            ret.append('disconnected')
            break
        j += 1
    else:
        ret.append('connected')
        ret.append(Param[p1_idx])
        ret.append(Param[p2_idx])
    return ret


###################################################################



open_logger('generate_introduction.log')

random.seed(102)


#################################################################
####### numerical results postprocessing ########################

log_debug('\n------------------------')
log_debug('post-processing\n')

filename = 'output_1.txt'

A_full, A_num, Param  = read_calculation_output2(filename)

log_debug(A_full, 'A_full', std_out=True)
log_debug(A_num, 'A_num', std_out=True)
log_debug(Param, 'Param', std_out=True)


######################################
# analyse potential sign


potential_sign = A_full['potential']

potential_signature = find_true_interval(potential_sign == '+', Param)
log_debug(potential_signature, 'interval of positive-definite potential', std_out=True)

potential_negative_signature = find_true_interval(potential_sign == '-', Param)
log_debug(potential_negative_signature, 'interval of non-positive-definite potential', std_out=True)


######################################
# analyse potential extrema

num_extrema = A_num['n_extr']
num_extrema_array = np.array(num_extrema.values, dtype=int)

# clip number of extrema to 2 (all num_extrema>1 belong to the same category)
num_extrema_array[num_extrema_array>1] = 2
num_extrema_unique = np.unique(num_extrema_array)

log_debug(num_extrema_unique, 'num_extrema_unique', std_out=True)

potential_extrema_signature = []
for n in num_extrema_unique:
    potential_extrema_signature.append(find_true_interval(num_extrema == n, Param))

log_debug(potential_extrema_signature, 'potential_extrema_signature', std_out=True)




# list of parameter values, where time-domain values differ more than 10% from WKB values

# data = A_num.values
# N = data.shape[0]
#
# w_re = data[:, 3]
# w_im = data[:, 5]
# t_re = data[:, 4]
# t_im = data[:, 6]
# r1 = np.sqrt(w_re**2 + w_im**2)
# r2 = np.sqrt(t_re**2 + t_im**2)
# d = np.sqrt( (t_re - w_re)**2 + (t_im - w_im)**2)
# rel_d = 2. * d / (r1 + r2 + 1e-12)
# q_points = (rel_d > 0.001)
#
#




####################################################
########## narrative ###############################

gen = Text_generator()
s = Sections()
vars = dict()


s.read_from_file('../templates/sections_results.txt')



#############################
#### potential signature ####

section_potential_signature = ''
# signature: {'connected', 'disconnected', 'all-true', 'all-false'}
if potential_signature[0] == 'all-true':
    section_potential_signature = 'potential_positive_definite_all'
elif potential_signature[0] == 'all-false':
    section_potential_signature = 'potential_non-positive_definite_all'
elif potential_signature[0] == 'connected':
    vars['_potential_positive_range_1'] = potential_signature[1]
    vars['_potential_positive_range_2'] = potential_signature[2]
    section_potential_signature = 'potential_positive_connected'
elif potential_signature[0] == 'disconnected':
    if potential_negative_signature[0] == 'connected':
        section_potential_signature = 'potential_negative_connected'
        vars['_potential_negative_range_1'] = potential_negative_signature[1]
        vars['_potential_negative_range_2'] = potential_negative_signature[2]
    else:
        section_potential_signature = 'potential_disconnected'
else:
    raise RuntimeError('case not implemented: {}'.format(potential_signature[0]))

p = s.get_section(section_potential_signature)
log_debug(p, 'section')
t = gen.generate_block(p, vars)
log_debug(t, 'generated text:\n', std_out=True)



#############################
#### potential extrema ######

# num_extrema_unique = {0, 1, 2}
#

# potential_extrema_signature = [
#   ['disconnected']
#   ['connected', a, b]
#   [''all-true']
#   ['all-false']
# ]


sections_potential_extrema = []
if num_extrema_unique.size == 1:
    # case 1:   same number of extrema for all params
    if num_extrema_unique[0] == 0:
        sections_potential_extrema = ['potential_0_extrema_all']
    elif num_extrema_unique[0] == 1:
        sections_potential_extrema = ['potential_1_extrema_all']
    else:
        sections_potential_extrema = ['potential_many_extrema_all']

elif num_extrema_unique.size == 2:
    # case 2:   two different numbers of extrema for different params
    if potential_extrema_signature[0][0] == 'connected':
        vars['_extrema_range_1'] = potential_extrema_signature[0][1]
        vars['_extrema_range_2'] = potential_extrema_signature[0][2]
        if num_extrema_unique[0] == 0:
            sections_potential_extrema.append('potential_0_extrema_ranged')
        if num_extrema_unique[0] == 1:
            sections_potential_extrema.append('potential_1_extrema_ranged')

        if num_extrema_unique[1] == 1:
            sections_potential_extrema.append('potential_1_extrema_continued')
        if num_extrema_unique[1] > 1:
            sections_potential_extrema.append('potential_many_extrema_continued')

    elif potential_extrema_signature[1][0] == 'connected':
        vars['_extrema_range_1'] = potential_extrema_signature[1][1]
        vars['_extrema_range_2'] = potential_extrema_signature[1][2]

        if num_extrema_unique[1] == 1:
            sections_potential_extrema.append('potential_1_extrema_ranged')
        if num_extrema_unique[1] > 1:
            sections_potential_extrema.append('potential_many_extrema_ranged')

        if num_extrema_unique[0] == 0:
            sections_potential_extrema.append('potential_0_extrema_continued')
        if num_extrema_unique[0] == 1:
            sections_potential_extrema.append('potential_1_extrema_continued')

    else:
        if num_extrema_unique[0] == 0 and num_extrema_unique[1] == 1:
            sections_potential_extrema.append('potential_01_extrema_disconnected')
        if num_extrema_unique[0] == 0 and num_extrema_unique[1] == 2:
            sections_potential_extrema.append('potential_02_extrema_disconnected')
        if num_extrema_unique[0] == 1 and num_extrema_unique[1] == 2:
            sections_potential_extrema.append('potential_12_extrema_disconnected')
else:
    # case 3:   three different numbers {0, 1, 2} of extrema for different params
    if potential_extrema_signature[0][0] == 'connected':
        vars['_extrema_range_1'] = potential_extrema_signature[0][1]
        vars['_extrema_range_2'] = potential_extrema_signature[0][2]
        sections_potential_extrema.append('potential_0_extrema_ranged')
        sections_potential_extrema.append('potential_12_extrema_continued')
    elif potential_extrema_signature[1][0] == 'connected':
        vars['_extrema_range_1'] = potential_extrema_signature[1][1]
        vars['_extrema_range_2'] = potential_extrema_signature[1][2]
        sections_potential_extrema.append('potential_1_extrema_ranged')
        sections_potential_extrema.append('potential_02_extrema_continued')
    elif potential_extrema_signature[2][0] == 'connected':
        vars['_extrema_range_1'] = potential_extrema_signature[2][1]
        vars['_extrema_range_2'] = potential_extrema_signature[2][2]
        sections_potential_extrema.append('potential_2_extrema_ranged')
        sections_potential_extrema.append('potential_01_extrema_continued')
    else:
        sections_potential_extrema.append('potential_012_disconnected')



for s_name in sections_potential_extrema:
    log_debug(s_name, 's_name', std_out=True)
    p = s.get_section(s_name)
    log_debug(p, 'section')
    t = gen.generate_block(p, vars)
    log_debug(t, 'generated text:\n', std_out=True)








