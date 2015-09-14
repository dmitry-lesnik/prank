import numpy as np
import logging
import pandas as pd
import time

# EPSILON = 1.e4 * np.finfo(float).eps   #  ~ 2.2e-12
# PRINT_LOG_DEBUG = False  # duplicate log to the standard output
# PRINT_LOG_INFO = True   # duplicate log to the standard output

_LOG_LEVEL = 0
_GLOBAL_INDENT = 0

def get_global_indent():
    return _GLOBAL_INDENT

def _get_indent(tab=0, with_global=True):
    if with_global:
        tab += _GLOBAL_INDENT
    ind = ''.join(['\t' for _i_ in range(tab)])
    return ind


def _tab_to_str(tab):
    ind = ''.join(['\t' for _i_ in range(tab)])
    return ind


def log_indent_inc(num=1):
    global _GLOBAL_INDENT
    _GLOBAL_INDENT += num
    _GLOBAL_INDENT = max(_GLOBAL_INDENT, 0)


def log_indent_dec(num=1):
    global _GLOBAL_INDENT
    _GLOBAL_INDENT -= num
    _GLOBAL_INDENT = max(_GLOBAL_INDENT, 0)



def open_logger(log_file_name="logfile.log", log_dir_name=None):
    if log_dir_name == "default":
        log_dir_name = "/home/dima/work/sandboxes/git_projects/sal/src/"
    if log_dir_name is not None:
        log_file_path = log_dir_name + '/' + log_file_name
    else:
        log_file_path = log_file_name
    logging.basicConfig(filename=log_file_path, filemode='w', format='%(message)s', level=logging.DEBUG)



def log_info(s, key='', tab=0):
    log_debug(s, key, tab, std_out=True)


def log_debug(s, key='', tab=0, std_out=False, level=0):
    if _LOG_LEVEL< level:
        return
    message = _get_indent(tab, with_global=True)
    global_tab = _GLOBAL_INDENT
    if key is not '':
        message += (key + ': ')
    if isinstance(s, tuple) and False:
        for idx, si in enumerate(s):
            log_debug(to_str(si, tab + global_tab, first_tab=False), key + "_" + str(idx), tab, std_out)
        return None
    else:
        message += to_str(s, tab + global_tab, first_tab=False)

    logging.debug(message)
    if std_out:
        print message


def to_str(x, tab=0, first_tab=True):

    # import core.svector

    apply_for_tuples = True
    apply_for_lists = True
    message = ''
    ind = _tab_to_str(tab)
    if first_tab:
        message += ind
    if isinstance(x, pd.tseries.index.DatetimeIndex):
        message += timeindex2str(x, tab, first_tab=False)
    elif isinstance(x, np.ndarray):
        message += array2str(x, tab, first_tab=False)
    elif isinstance(x, tuple) and apply_for_tuples:
        message += "[tuple]({}):".format(len(x))
        for idx, xi in enumerate(x):
            message += "\n{}\t({}):\t{}".format(ind, idx, to_str(xi, tab+1, first_tab=False))
    elif isinstance(x, list) and apply_for_lists:
        message += "[list]({}):".format(len(x))
        for idx, xi in enumerate(x):
            message += "\n{}\t({}):\t{}".format(ind, idx, to_str(xi, tab+1, first_tab=False))
    elif isinstance(x, float):
        message += "{:.10g}".format(x)
    elif isinstance(x, dict):
        message += dict2str(x, tab, first_tab=False)
    elif isinstance(x, pd.Series):
        message += series2str(x, tab, first_tab=False)
    elif type(x).__name__ in ['SRow', 'CVector', 'EVector', 'SMatrix', 'iMatrix', 'pRow', 'pVector']:
        message += x.to_str(tab=tab, first_tab=first_tab)
    else:
        message += str(x)

    return message


def array2str(x, tab=0, first_tab=True):
    ndim = x.ndim
    ind = _get_indent(tab, with_global=False)
    p = ''
    if ndim == 1 or (ndim == 2 and x.shape[0] == 1):
        if first_tab:
            p = ind
        p += str(x.shape) + ":\t"
        if x.dtype.name == 'float64' or x.dtype.name == 'float32':
            p += "\t".join(["{:.11g}".format(i) for i in x.reshape((-1,))])
        elif x.dtype.name == 'int64' or x.dtype.name == 'int32':
            p += "\t".join(["{}".format(i) for i in x.reshape((-1,))])
        else:
            p += '\n'
            p += "\n".join(["{}\t{}".format(ind, to_str(i, first_tab=False)) for i in x.reshape((-1,))])
    else:
        if first_tab:
            p = ind
        p += str(x.shape) + ":\n"
        p += "\n".join([array2str(i, tab+1) for i in x])
    return p

def timeindex2str(x, tab=0, first_tab=True):
    ind = _get_indent(tab)
    p = ''
    if first_tab:
        p = ind
    p += "TimeIndex " + str(x.shape) + ' { '
    p += "\t".join(["{},".format(i) for i in x]) + '  }'
    return p

def series2str(x, tab=0, first_tab=True):
    ind = _get_indent(tab)
    p = ''
    if first_tab:
        p = ind
    p += "Series:"
    T = x.index
    V = x.values

    N = V.size
    for i in range(N):
        p += '\n' + ind + "\t{}  \t{}".format(T[i], V[i])
    return p


def dict2str(D, tab=0, first_tab=True):
    ind = _get_indent(tab+1, with_global=False)
    p = ''
    if first_tab:
        p += ind
    p += "[dict]({}):".format(len(D))
    for key, value in D.items():
        # p += '\n' + ind + key + ":\t" + to_str(value, tab+1, first_tab=False)
        p += '\n{}{:20s}:\t{}'.format(ind, str(key), to_str(value, tab+1, first_tab=False))
    return p


def log_function_enter(message='', silent=False):
    if silent:
        log_debug('Calling  ' + message)
    else:
        log_debug('{ Entering  ' + message + ' ')
    log_indent_inc()
    return time.time()


def log_function_exit(message='', silent=False, start_time=None):
    log_indent_dec()
    if not silent:
        if start_time is None:
            log_debug('} - Exit  ' + message + ' ---')
        else:
            end_time = time.time()
            log_debug('} - Exit  ' + message + ';  elapsed time = {}'.format(end_time-start_time))


# ---------  decorator (log function entry and exit) -----------------

def log(name=None, level=0):
    """
    Wrap function into entry and exit messages
    { enter function
    } exit function
    """
    if _LOG_LEVEL < level:
        def wrapped(function):
            return function
        return wrapped
    else:
        def wrapped(function, name=name):
            if name is None:
                name = function.__name__
            else:
                name = name + '.' + function.__name__

            def f(*args, **kwargs):
                start_time = log_function_enter(name)
                r = function(*args, **kwargs)
                log_function_exit(name, start_time=start_time)
                return r

            return f
        return wrapped

# ---------  decorator (log function entry) -----------------

def logs(name=None, level=0):
    """
    Print function entry message and increase indentation
    """
    if _LOG_LEVEL < level:
        def wrapped(function):
            return function
        return wrapped
    else:
        def wrapped(function, name=name):
            if name is None:
                name = function.__name__
            else:
                name = name + '.' + function.__name__
            def f(*args, **kwargs):
                log_debug("Calling {}".format(name))
                log_indent_inc()
                r = function(*args, **kwargs)
                log_indent_dec()
                return r
            return f
        return wrapped



#=============================================================================================

if __name__ == "__main__":

    import numpy as np

    log_file_name = "helpers.log"
    open_logger(log_file_name)
    log_debug("testing logger\n\n")

    x = np.ones((2,3,4), dtype=float)
    log_debug(x, "array ones(2,3,4)")
    log_debug('--------------------------')

    x = np.ones((3,1,4), dtype=float)
    log_debug(x, "array ones(3,1,4)", tab=3)
    log_debug('--------------------------')



    y = (2, 1.2345566, np.arange(5, dtype=float))
    log_debug(y, "tuple", tab=2)
    log_debug('--------------------------')


    y = dict(
        key1=1,
        key2=1.234567,
        key3=np.linspace(0., 1., 5),
        key4=(2, 1.2345566, np.arange(5, dtype=float))
    )
    log_debug(y, "d", tab=2)
    log_debug('--------------------------')

    log_indent_inc(3)
    z = (np.pi,
         dict(
             key1=1,
             key2=1.234567,
             key3=np.linspace(0., 1., 5),
             key4=(2, 1.2345566, np.eye(5, dtype=float))
            ),
         [2, 1.2345566, np.eye(5, dtype=float)]
         )
    log_debug(z, "mega-structure")
    log_indent_dec(3)
    log_debug('--------------------------')
    log_indent_dec()
    log_debug('end of test')









