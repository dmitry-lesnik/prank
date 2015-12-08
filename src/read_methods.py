import numpy as np
import pandas as pd
from helpers import *



def read_synonyms(filename):
    """
    Returns:
    ret:    list of lists
        each line is a list of words
    """
    f = open(filename)
    x = f.readlines()
    ret = []
    for line in x:
        z = line.split(',')
        for i in range(len(z)):
            z[i] = z[i].strip(' \n')
        if z != ['']:
            ret.append(z)
    f.close()
    return ret


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
        z = line.strip(' \n')
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
    A = pd.read_csv(filename, delimiter=',\s*')
    a, b, v, v_a0, v_b0 = analyse_output(A)
    return A, a, b, v, v_a0, v_b0


def analyse_output(A):
    """
    A: DataFrame
        2D table with calculation output

        columns 1 and 2 - parameters
        columns 3 and 4 - Re(omega) and Im(omega) by WKB method
        columns 5 and 6 - Re(omega) and Im(omega) by time domain method

    """
    a_full = A.values[:,0].astype(float, copy=False)
    a = np.unique(a_full)
    log_debug(a, 'parameter a')
    a_name = A.columns[0]
    log_debug(a_name, 'a_name')

    b_full = A.values[:,1].astype(float, copy=False)
    b = np.unique(b_full)
    log_debug(b, 'parameter b')
    b_name = A.columns[1]
    log_debug(b_name, 'b_name')

    v = A.values[:, 2:]

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

    return a, b, v, v_a0, v_b0


if __name__ == '__main__':

    open_logger('read_methods.log')

    # filename = 'calculation_output.txt'
    #
    # A = read_calculation_output(filename)
    #
    # log_debug(A, 'A', std_out=True)
    #
    # a, b, v, v_a0, v_b0 = analyse_output(A)
    #
    #
    # print b
    #
    #
    # ret = read_templates('sentence_templates.txt')
    #
    # log_debug(ret, 'ret', std_out=True)



    filename = 'calculation_output2.txt'

    A, a, b, v, v_a0, v_b0 = read_calculation_output(filename)

    log_debug(A, 'A', std_out=True)









