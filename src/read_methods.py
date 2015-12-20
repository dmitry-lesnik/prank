import numpy as np
import pandas as pd
from helpers import *




def read_templates(filename):
    """
    Returns
    -------
    ret:    dict(key, list(string))
        each entry is a list of equivalent sentences
        key is a string with prefix **

    """
    f = open(filename)
    lines = f.readlines()
    ret = dict()
    key = 'key_missing'
    for line in lines:
        z = line.strip(' \n\r')
        if len(z) == 0:
            pass
        elif z[0] == '#':
            # skip comment
            pass
        elif z[0:2] == '**':
            # this is a new entry
            key = z
            ret[key] = []
        else:
            # this is the entry continuation
            if key == 'key_missing':
                raise RuntimeError('sentence_templates should start with a key with a prefix **')
            ret[key].append(z)

    f.close()
    return ret



def read_calculation_output(filename='calculation_output.txt'):

    A_full = pd.read_csv(filename, delimiter=',\s*')
    A_num = pd.read_csv(filename, delimiter=',\s*', na_values=['stable', 'error', '-'])


    a_full = A_num.values[:,0].astype(float, copy=False)
    a = np.unique(a_full)
    log_debug(a, 'parameter a')
    a_name = A_num.columns[0]
    log_debug(a_name, 'a_name')

    b_full = A_num.values[:,1].astype(float, copy=False)
    b = np.unique(b_full)
    log_debug(b, 'parameter b')
    b_name = A_num.columns[1]
    log_debug(b_name, 'b_name')

    v = A_num.values[:, 2:]

    # mean value of parameter a
    n = a.size
    a0 = a[n/2]
    log_debug(a0, 'a0')
    # mean value of parameter b
    n = b.size
    b0 = b[n/2]
    log_debug(b0, 'b0')

    # slice of values corresponding to mean value of parameters a and b

    filter = (a_full == a0)
    v_a0 = v[filter, :]
    log_debug(v_a0, 'v_a0')

    filter = (b_full == b0)
    v_b0 = v[filter, :]
    log_debug(v_b0, 'v_b0')

    ret = dict()
    ret['A_full'] = A_full
    ret['A_num'] = A_num
    ret['param_a'] = a
    ret['param_b'] = b
    ret['values_full'] = v
    ret['values_slice_a0'] = v_a0
    ret['values_slice_b0'] = v_b0
    return ret

def read_calculation_output2(filename):

    A_full = pd.read_csv(filename,
                         delimiter=',',
                         skipinitialspace=True)
    A_num = pd.read_csv(filename,
                        delimiter=',\s*',
                        na_values=['stable', 'error', '-', '+'],
                        skipinitialspace=True)

    good_rows = -np.isnan(A_num.P)
    A_num = A_num[good_rows]
    A_full = A_full[good_rows]
    Param = A_num.values[:,0].astype(float, copy=False)

    return A_full, A_num, Param



if __name__ == '__main__':

    open_logger('read_methods.log')



    filename = 'output_1.txt'

    A_full, A_num, Param = read_calculation_output2(filename)

    log_debug(A_full, 'A_full', std_out=True)
    log_debug(A_num, 'A_num', std_out=True)
    log_debug(Param, 'Param', std_out=True)










