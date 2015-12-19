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
        'all-true' if all x == true
        'all-false' if all x == false
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


def template_from_intervals(intervals, signature, var):
    N = len(signature)
    if N < 1:
        p = []
    elif N == 1:
        if signature[0] == 1:
            p = ['**monotonicity_+|{}'.format(var)]
        else:
            p = ['**monotonicity_-|{}'.format(var)]
    elif N == 2:
        if signature[0] == 1 and signature[1] == -1:
            p = ['**monotonicity_+-|{}|{:.2g}'.format(var, intervals[0][2])]
        elif signature[0] == -1 and signature[1] == 1:
            p = ['**monotonicity_-+|{}|{:.2g}'.format(var, intervals[0][2])]
        else:
            raise RuntimeError('incorrectly detected monotonicity')
    elif N > 2:
        p = []
    return p


###################################################################


open_logger('generate_introduction.log')

random.seed(102)


#####################################################################
############## numerical results postprocessing #####################
#####################################################################


log_debug('\n------------------------')
log_debug('post-processing\n')

filename = 'output_1.txt'

A_full, A_num, Param  = read_calculation_output2(filename)

log_debug(A_full, 'A_full', std_out=True)
log_debug(A_num, 'A_num', std_out=True)
log_debug(Param, 'Param', std_out=True)


#########################################
######## analyse potential sign #########


potential_sign = A_full['potential']

potential_signature = find_true_interval(potential_sign == '+', Param)
log_debug(potential_signature, 'interval of positive-definite potential', std_out=True)

potential_negative_signature = find_true_interval(potential_sign == '-', Param)
log_debug(potential_negative_signature, 'interval of non-positive-definite potential', std_out=True)


#########################################
###### analyse potential extrema ########


# output: num_extrema_unique - list of unique extrema numbers
# output: potential_extrema_signature - for each unique extrema number this is a list containing
#   ['disconnected']
#   ['connected', a, b]
#   ['all-true']
#   ['all-false']
#             where [a, b] is the range of the corresponding number of extrema


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


######################################
# analyse wkb_re

wkb_re = A_num['wkb_re']
log_debug(wkb_re, 'wkb_re')

wkb_re_all_nan = np.isnan(wkb_re).all()
wkb_re_some_nan = np.isnan(wkb_re).any()

intervals_wkb_re = []
signature_wkb_re = []
if not wkb_re_some_nan:
    intervals_wkb_re, signature_wkb_re = get_full_analysis(Param, wkb_re.values)

log_debug(intervals_wkb_re, 'intervals_wkb_re', std_out=True)


######################################
# analyse wkb_im

wkb_im = A_num['wkb_im']
log_debug(wkb_im, 'wkb_im')

wkb_im_all_nan = np.isnan(wkb_im).all()
wkb_im_some_nan = np.isnan(wkb_im).any()

intervals_wkb_im = []
signature_wkb_im = []
if not wkb_im_some_nan:
    intervals_wkb_im, signature_wkb_im = get_full_analysis(Param, wkb_im.values)

log_debug(intervals_wkb_im, 'intervals_wkb_im', std_out=True)


######################################
# analyse td_re

td_re = A_num['td_re']
log_debug(td_re, 'td_re')

td_re_all_nan = np.isnan(td_re).all()
td_re_some_nan = np.isnan(td_re).any()

intervals_td_re = []
signature_td_re = []
if not td_re_some_nan:
    intervals_td_re, signature_td_re = get_full_analysis(Param, td_re.values)

log_debug(intervals_td_re, 'intervals_td_re', std_out=True)


######################################
# analyse td_im

td_im = A_num['td_im']
log_debug(td_im, 'td_im')

td_im_all_nan = np.isnan(td_im).all()
td_im_some_nan = np.isnan(td_im).any()

intervals_td_im = []
signature_td_im = []
if not td_im_some_nan:
    intervals_td_im, signature_td_im = get_full_analysis(Param, td_im.values)

log_debug(intervals_td_im, 'intervals_td_im', std_out=True)


################################################
# analyse td_im for stable and unstable regions

unstable_td_signature = find_true_interval(td_im > 0, Param)
log_debug(unstable_td_signature, 'unstable_td_signature', std_out=True)

stable_td_signature = find_true_interval(A_full['td_im'] == 'stable', Param)
log_debug(stable_td_signature, 'stable_td_signature', std_out=True)


####################################################################
## list of parameter values, where time-domain values differ more than 10% from WKB values
## entries with td_im > 0 are excluded

w_re = A_num['wkb_re'].values
w_im = A_num['wkb_im'].values
t_re = A_num['td_re'].values
t_im = A_num['td_im'].values

t_re[t_im>0] = np.nan # exclude unstable region
t_im[t_im>0] = np.nan # exclude unstable region

d_re = np.abs(w_re - t_re)
d_im = np.abs(w_im - t_im)

rel_d_re = d_re / np.max(np.vstack([w_re, t_re]), axis=0)
rel_d_im = d_im / np.max(np.vstack([w_im, t_im]), axis=0)
max_rel_d = np.max(np.vstack([rel_d_re, rel_d_im]), axis=0)
log_debug(max_rel_d, 'max_rel_d', std_out=True)
large_diff_points = (max_rel_d > 0.01)
large_diff_points = pd.Series(large_diff_points)
log_debug(large_diff_points, 'large_diff_points', std_out=True)

large_difference_signature = find_true_interval(large_diff_points, Param)


########################################################################
############################## narrative ###############################
########################################################################


gen = Text_generator()
s = Sections()
vars = dict()


s.read_from_file('../templates/sections_introduction.txt')
s.read_from_file('../templates/sections_results.txt', reset=False)

full_text = []
full_text += s.get_section('introduction_2')
full_text += s.get_section('results_1')


###############################################################
#################### potential signature ######################

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



log_debug('\nanalysis of the potential signature:', std_out=True)
p = s.get_section(section_potential_signature)
log_debug(p, 'section')
t = gen.generate_block(p, vars)
log_debug(t, 'generated text:\n', std_out=True)

full_text += p


############################################################
################### potential extrema ######################

# possible values of num_extrema_unique = {0, 1, 2}
#

# potential_extrema_signature = [
#   ['disconnected']
#   ['connected', a, b]
#   ['all-true']
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



log_debug('\nanalysis of the potential extrema:', std_out=True)
for s_name in sections_potential_extrema:
    log_debug(s_name, 'section_name', std_out=True)
    p = s.get_section(s_name)
    log_debug(p, 'section')
    t = gen.generate_block(p, vars)
    log_debug(t, 'generated text:\n', std_out=True)
    full_text += p




log_debug('\nanalysis of the omega:', std_out=True)


################### wkb_re ######################
if not wkb_re_some_nan:
    p = template_from_intervals(intervals_wkb_re, signature_wkb_re, var='$Re \omega$')
    if p != []:
        t = gen.generate_block(p, vars)
        log_debug(t, 'generated text:\n', std_out=True)
        full_text += p


################### wkb_im ######################
if not wkb_im_some_nan:
    p = template_from_intervals(intervals_wkb_im, signature_wkb_im, var='$Im \omega$')
    if p != []:
        t = gen.generate_block(p, vars)
        log_debug(t, 'generated text:\n', std_out=True)
        full_text += p


################### td_re ######################
if not td_re_some_nan:
    p = template_from_intervals(intervals_td_re, signature_td_re, var='$Re \omega$ calculated by time domain simulation')
    if p != []:
        t = gen.generate_block(p, vars)
        log_debug(t, 'generated text:\n', std_out=True)
        full_text += p


################### td_im ######################
if not td_im_some_nan:
    p = template_from_intervals(intervals_td_im, signature_td_im, var='$Im \omega$ calculated by time domain simulation')
    if p != []:
        t = gen.generate_block(p, vars)
        log_debug(t, 'generated text:\n', std_out=True)
        full_text += p


################# instability region #####################

section_instability_td = ''
if unstable_td_signature[0] != 'all-false':
    if unstable_td_signature[0] == 'all-true':
        section_instability_td = 'instability_td_all'
    elif unstable_td_signature[0] == 'connected':
        section_instability_td = 'instability_td_range'
        vars['_instability_range_1'] = unstable_td_signature[1]
        vars['_instability_range_2'] = unstable_td_signature[2]
    elif unstable_td_signature[0] == 'disconnected':
        section_instability_td = 'instability_td_disconnected'

log_debug(section_instability_td, 'section_instability_td', std_out=True)
p = s.get_section(section_instability_td)
log_debug(p, 'section')
t = gen.generate_block(p, vars)
log_debug(t, 'generated text:\n', std_out=True)
full_text += p




################# stability region #####################

section_stability_td = ''
if stable_td_signature[0] != 'all-false':
    if stable_td_signature[0] == 'all-true':
        section_stability_td = 'stability_td_all'
    elif stable_td_signature[0] == 'connected':
        section_stability_td = 'stability_td_range'
        vars['_stability_range_1'] = stable_td_signature[1]
        vars['_stability_range_2'] = stable_td_signature[2]
    elif stable_td_signature[0] == 'disconnected':
        section_stability_td = 'stability_td_disconnected'

log_debug(section_stability_td, 'section_stability_td', std_out=True)
p = s.get_section(section_stability_td)
log_debug(p, 'section')
t = gen.generate_block(p, vars)
log_debug(t, 'generated text:\n', std_out=True)
full_text += p


################# large difference region #####################


section_large_difference = ''
if large_difference_signature[0] != 'all-false':
    if large_difference_signature[0] == 'all-true':
        section_large_difference = 'large_difference_all'
    elif large_difference_signature[0] == 'connected':
        section_large_difference = 'large_difference_range'
        vars['_large_difference_range_1'] = large_difference_signature[1]
        vars['_large_difference_range_2'] = large_difference_signature[2]
    elif large_difference_signature[0] == 'disconnected':
        section_large_difference = 'large_difference_disconnected'

log_debug(section_large_difference, 'section_large_difference', std_out=True)
p = s.get_section(section_large_difference)
log_debug(p, 'section')
t = gen.generate_block(p, vars)
log_debug(t, 'generated text:\n', std_out=True)
full_text += p









##################################################################




log_debug(full_text, 'full_text')
gen.write_to_file('generated_text.txt', full_text, vars )









##################################################################


