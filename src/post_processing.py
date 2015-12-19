import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from helpers import *


def grad1d(y, dx):
    """
    y :     array(N)
        y-values, supposed to be equidistant

    dx :    float
        grid step

    Returns
    -------
    g :     array(N)
        gradient g = y'(x) calculated at the same points of x
    """
    g = np.zeros(y.size, dtype=float)
    g[1:-1] = (y[2:] - y[:-2]) / dx / 2.
    g1 = (y[1] - y[0]) / dx
    g2 = (y[2] - y[1]) / dx
    g[0] = (3. * g1 - g2) / 2.
    g1 = (y[-1] - y[-2]) / dx
    g2 = (y[-2] - y[-3]) / dx
    g[-1] = (3. * g1 - g2) / 2.
    return g


@log('')
def filtered_gradient(x, y, n1):
    """
    x :     array
        x-values

    y :     array
        y-values

    n1 :    int
        new number of x-points

    Returns
    -------
    grad_x :    array[float](N)
        new x-values

    grad_y :    array[float](N)
        values of y'(x) at new x-values

    grad_sign : array[int = -1, 0, 1](N)
        gradient sign with epsilon-precision,
        sticks to zero if gradient is epsilon-small

    """
    n = x.size
    dx = (x[-1] - x[0]) / (n - 1.)
    grad = grad1d(y, dx)
    grad2 = grad1d(grad, dx)
    grad3 = grad1d(grad2, dx)

    # estimate epsilon
    max_d3y = abs(grad3).mean()
    eps = 0.5 * (max_d3y * dx ** 3)

    # re-sample gradient
    f = interp1d(x, grad, kind='quadratic')

    grad_x = np.linspace(x[0], x[-1], n1)
    grad_y = f(grad_x)

    grad_sign = np.zeros(n1, dtype=int)
    grad_sign[grad_y > 0] = 1
    grad_sign[grad_y < 0] = -1
    grad_sign[abs(grad_y) < eps] = 0

    # log_debug(x, 'x')
    # log_debug(y, 'y')
    # log_debug(eps, 'epsilon estimate')
    # log_debug(grad_x, 'grad_x')
    # log_debug(grad_y, 'grad_y')
    # log_debug(grad_sign, 'grad_sign')

    return grad_x, grad_y, grad_sign


def find_regions(grad_x, grad_sign):
    """
    Returns
    -------
    limits_x :  array(N+1)
        limits of monotonicity intervals

    signs :     array[int = -1, 1](N)
        signs on each interval
        signs do not include zeros, only -1 and 1
    """
    n = grad_x.size
    limits_x = []
    signs = []

    i = 0
    while grad_sign[i] == 0:
        i += 1
        if i == n:
            return limits_x, signs
    if i == 0:
        limits_x.append(grad_x[i])
    else:
        limits_x.append(grad_x[i - 1])

    while i < n:
        signs.append(grad_sign[i])
        while grad_sign[i] == signs[-1]:
            i += 1
            if i == n:
                limits_x.append(grad_x[i - 1])
                return limits_x, signs
        x1 = grad_x[i - 1]
        x2 = grad_x[i]
        if grad_sign[i] == 0:
            while i < n and grad_sign[i] == 0:
                i += 1
            if i == n:
                x1 = x2
            else:
                x2 = grad_x[i]
        limits_x.append(0.5 * (x1 + x2))
    return limits_x, signs


def monotonous_intervals(limits_x, signs, use_inflection=False):

    """
    Returns
    -------

    intervals :     list[tuple[sign, from, to]]
        each tuple is an interval of monotonicity or inflection point;
    """

    intervals = []
    n = len(signs)
    i = 0
    while i < n:
        s = signs[i]
        a = limits_x[i]
        b = limits_x[i + 1]
        if i < n - 1:
            if signs[i + 1] == s:
                i += 1
                b = limits_x[i + 1]
                intervals.append((s, a, b))
                if use_inflection:
                    intervals.append((0, limits_x[i], limits_x[i]))
            else:
                intervals.append((s, a, b))
        else:
            intervals.append((s, a, b))
        i += 1
    return intervals


def get_signature(intervals):
    """
    Returns
    -------
    signature : list[int = -1, 1]
        sequence of gradient signs
    """

    sig = []
    for itv in intervals:
        if itv[0] != 0:
            sig.append(itv[0])
    return sig


def get_full_analysis(x, y, use_inflection=False):
    n = x.size
    n1 = 10 * n
    grad_x, grad_y, grad_sign = filtered_gradient(x, y, n1)
    limits_x, signs = find_regions(grad_x, grad_sign)
    intervals = monotonous_intervals(limits_x, signs, use_inflection)
    sig = get_signature(intervals)
    return intervals, sig




def narrative(intervals):
    n = len(intervals)

    n_inflection = 0
    for t in intervals:
        if t[0] == 0:
            n_inflection += 1

    n_monotonicity = n - n_inflection

    sentences = []

    if n_inflection > 0:
        sentences.append(
            'In the considered range Omega has {} intervals of monotonicity and {} points of inflection.'.format(
                n_monotonicity, n_inflection))
    else:
        sentences.append(
            'In the considered range Omega has {} intervals of monotonicity.'.format(
                n_monotonicity))

    for t in intervals:
        if t[0] == -1:
            sentences.append('In the range [{:.3g}, {:.3g}] Omega is decreasing.'.format(t[1], t[2]))
        elif t[0] == 1:
            sentences.append('In the range [{:.3g}, {:.3g}] Omega is increasing.'.format(t[1], t[2]))
        elif t[0] == 0:
            sentences.append('At X={:.3g} Omega has a point of inflection.'.format(t[1]))

    return sentences


if __name__ == "__main__":

    open_logger('postprocessing.log')

    # input
    x = np.linspace(0., 5., 21)
    y = (x - 2.) ** 3 + 0.01 * x - 3. * np.exp(2. * (x - 4.))

    n = x.size
    n1 = 10 * n

    grad_x, grad_y, grad_sign = filtered_gradient(x, y, n1)

    limits_x, signs = find_regions(grad_x, grad_sign)
    log_debug(limits_x, 'limits_x', std_out=True)
    log_debug(signs, 'signs', std_out=True)

    intervals = monotonous_intervals(limits_x, signs)

    log_debug(intervals, 'intervals', std_out=True)


    # plt.plot(x,y)
    # plt.plot(grad_x, grad_y)
    # plt.grid(True)
    # plt.show()


    sig = get_signature(intervals)

    log_debug(sig, 'signature', std_out=True)

    sentences = narrative(intervals)
    for s in sentences:
        log_debug(s, std_out=True)




