import numpy as np
import pandas as pd
from helpers import *



def read_calculation_output(filename='calculation_output.txt'):
    A = pd.read_csv(filename, delimiter=',\s*')
    return A



def analyse_output(A):
    """
    A: DataFrame
        2D table with calculation output

        columns 1 and 2 - parameters
        columns 3 and 4 - Re(omega) and Im(omega) by WKB method
        columns 5 and 6 - Re(omega) and Im(omega) by time domain method

    """
    a_full = A.values[:,0]
    a = np.unique(a_full)
    log_debug(a, 'parameter a')
    a_name = A.columns[0]
    log_debug(a_name, 'a_name')

    b_full = A.values[:,1]
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

    filename = 'calculation_output.txt'

    A = read_calculation_output(filename)

    log_debug(A, 'A', std_out=True)

    a, b, v, v_a0, v_b0 = analyse_output(A)


    print b












