import os

import pandas as pd
import numpy as np
from numpy.testing import (
    assert_allclose, assert_raises, run_module_suite, assert_equal
)
from mkl.interpolation import Interpolate1D

from gazprom.mt.pricing.models.lsmc import LeastSquaresMonteCarloEngine
from gazprom.mt.pricing.models.lsmc import StateObject

from gazprom.mt.pricing.models.lsmc.constraints import (
    InventoryConstraints,
    FlowConstraint,
    FlowConstraints
)
from gazprom.mt.pricing.models.lsmc.cashflows import (
    SwingCashflows,
    StorageCashflows,
)

from gazprom.mt.pricing.utils import AttrDict
from gazprom.mt.pricing.models.timeseries.multifactor import MultiFactorSimulator
from gazprom.mt.pricing.lib.financial import calcTe

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt



from log import *
from mu_bar import *
from intr_trigger_prices import calculate_trigger_prices, calculate_trigger_prices2





def rms(x):
    return np.sqrt( (x**2).mean() )


class Constraints(object):
    def __init__(self):
        self.max_inv = None
        self.min_inv = None
        self.start_inv = None
        self.end_inv = None
        self.inj_cost = None
        self.rel_cost = None
        self.rmin = None
        self.rmax = None

class ModelParams(object):
    def __init__(self):
        self.sigma_1 = None
        self.sigma_2 = None
        self.alpha_1 = None
        self.alpha_2 = None
        self.rho = None




def run_one_valuation_intrinsic(Te, fwd_array, cons):
    """
    Calculate intrinsic storage with simple constraints

    Parameters
    ----------
    Te  :       maturities

    fwd_array:

    cons :  Constraints
        (max_inv, min_inv, start_inv, end_inv, inj_cost, rel_cost, rmin, rmax)

    Returns
    -------

    """


    max_inv = cons.max_inv
    min_inv = cons.min_inv
    start_inv = cons.start_inv
    end_inv = cons.end_inv
    inj_cost = cons.inj_cost
    rel_cost = cons.rel_cost
    rmin = cons.rmin
    rmax = cons.rmax

    fc = FlowConstraint(rmin, rmax)
    flow_constraints = FlowConstraints(Te[0], fc)

    inv_constraints = InventoryConstraints(Te, start_inv, end_inv, max_inv, min_inv, [flow_constraints])
    state_object = StateObject(inv_constraints)

    product = LeastSquaresMonteCarloEngine(state_object,
                                           StorageCashflows(Te,
                                                            injection_cost=inj_cost,
                                                            withdrawal_cost=rel_cost))

    res = product.calc_intrinsic(fwd_array)
    TP = product.calc_trigger_price(res)
    return res, TP


def run_valuation_stochastic(val_date, dates, fwd_curve, cons, model_params, nsims=5000):
    """
    Calculate stochastic storage with simple constraints

    Parameters
    ----------
    Te  :       maturities

    fwd_array:

    cons :  tuple
        (max_inv, min_inv, start_inv, end_inv, inj_cost, rel_cost, rmin, rmax)

    Returns
    -------

    """

    max_inv = cons.max_inv
    min_inv = cons.min_inv
    start_inv = cons.start_inv
    end_inv = cons.end_inv
    inj_cost = cons.inj_cost
    rel_cost = cons.rel_cost
    rmin = cons.rmin
    rmax = cons.rmax

    sigma_1 = model_params.sigma_1
    sigma_2 = model_params.sigma_2
    alpha_1 = model_params.alpha_1
    alpha_2 = model_params.alpha_2
    rho = model_params.rho


    alphas = [alpha_1, alpha_2]
    sigmas = [sigma_1, sigma_2]

    Te = calcTe(val_date, dates)

    fc = FlowConstraint(rmin, rmax)
    flow_constraints = FlowConstraints(Te[0], fc)

    inv_constraints = InventoryConstraints(Te, start_inv, end_inv, max_inv, min_inv, [flow_constraints])
    state_object = StateObject(inv_constraints, num_levels=None)
    ls_solver_params = AttrDict(max_basis_order=4, solver='RegularisedLeastSquaresSolver')
    product = LeastSquaresMonteCarloEngine(state_object,
                                           StorageCashflows(Te, injection_cost=inj_cost, withdrawal_cost=rel_cost),
                                           ls_solver_params)


    seed = 123
    simulator = MultiFactorSimulator(alphas, sigmas, fwd_curve, rho, val_date, seed=seed)
    sims_bwd = simulator.simulate_spot(dates, nsims).values
    factors_bwd = simulator.stoch_factors
    sims_fwd = simulator.simulate_spot(dates, nsims).values
    factors_fwd = simulator.stoch_factors

    results = product.calc_extrinsic(sims_bwd, sims_fwd, factors_bwd, factors_fwd)
    cashflows = (results['cashflows'].spot + results['cashflows'].opex).sum(axis=0)
    value = cashflows.mean()
    value_std = cashflows.std()
    value_error = value_std/np.sqrt(nsims)

    return value, value_error




def test_trigger_prices():

    nTe = 100
    Te = 0.2 + np.arange(nTe, dtype=float) / 365.
    Te_ext = np.append(Te, Te[-1] + 1./365.)


    cons = Constraints()
    cons.max_inv = 15.
    cons.min_inv = 0.
    cons.start_inv = 0.0
    cons.end_inv = cons.min_inv
    cons.inj_cost = 2.
    cons.rel_cost = 1.
    cons.rmin = -1.
    cons.rmax = 1.
    # cons = (max_inv, min_inv, start_inv, end_inv, inj_cost, rel_cost, rmin, rmax)

    P = 1.3
    fwd_array = 20. + \
                5.0 * np.sin(-0.1 + np.linspace(0., 2.0 * np.pi * P, nTe)) + \
                np.linspace(0., 3., nTe)

    results, X = run_one_valuation_intrinsic(Te, fwd_array, cons)

    cashflows = results['cashflows'].spot + results['cashflows'].opex
    mean_cf = cashflows.mean(axis=1)
    value = mean_cf.sum()
    fd = mean_cf/fwd_array
    mean_actions = results.actions.mean(axis=1)
    # print "value = {}".format(value)
    # print "delta = {}".format(fd)
    # print "actions = {}".format(mean_actions)

    levels = np.empty(nTe+1, dtype=float)
    levels[0] = cons.start_inv
    levels[1:] = np.cumsum(mean_actions)

    print "levels = {} (num levels = {})".format(levels, levels.size)
    print "trigger price = {}".format(X)
    print "actions = ", mean_actions

    triggers = calculate_trigger_prices(X['dVdq'], Te, fwd_array, mean_actions, cons)

    # trigger_prices, trigger_prices_upper, trigger_prices_lower, tt_up, tt_down, fdot_up, fdot_down, F0_up, F0_down

    # trigger_prices_upper = trigger_prices + rel_cost
    # trigger_prices_low = trigger_prices - inj_cost
    # tr_m = []
    # tr_u = []
    # tr_l = []
    # for t in range(nTe):
    #     print "{}  of  {}, \tstarting level = {}\texercise = {}".format(t, nTe-1, levels[t], mean_actions[t])
    #     start_inv = levels[t]
    #     cons = (max_inv, min_inv, start_inv, end_inv, inj_cost, rel_cost, rmin, rmax)
    #     res, X = run_one_valuation_intrinsic(Te[t:], fwd_array[t:], cons)
    #     tr_m.append(X['dVdq'])
    #     tr_l.append(X['lower'])
    #     tr_u.append(X['upper'])
    # print 'trigger prices = {}'.format(np.asarray(tr_m))

    minor_sticks = []
    for t in range(nTe)[1:]:
        if mean_actions[t] != mean_actions[t-1]:
            minor_sticks.append(Te[t])

    import matplotlib.pyplot as plt

    plt.close('all')
    plt.figure()
    plt.plot(Te, fwd_array, 'o-', linewidth=1, markersize=2)
    plt.plot(Te_ext, levels)
    # plt.plot(Te, tr_m)
    # plt.plot(Te, tr_u)
    # plt.plot(Te, tr_l)
    plt.plot(Te, triggers.trigger_prices, 'o-', linewidth=1, markersize=2)
    plt.plot(Te, triggers.trigger_prices_upper, '--', linewidth=1)
    plt.plot(Te, triggers.trigger_prices_lower, '--', linewidth=1)
    # plt.legend(['fwd', 'level', 'tr_mean', 'tr_up', 'tr_low'], loc=4)
    plt.ylim(-1., 30.)

    ax = plt.gca()
    # start, end = ax.get_xlim()
    ax.xaxis.set_ticks(minor_sticks, minor=True)
    plt.grid(True)
    ax.xaxis.grid(True, which='minor')
    plt.show()



def setting_1():

    val_date = pd.datetime(2015, 1, 1)
    start_date = pd.datetime(2015, 1, 1)
    nTe = 300
    dates = pd.date_range(start_date, periods=nTe)

    cons = Constraints()
    cons.max_inv = 50.
    cons.min_inv = 0.
    cons.start_inv = 20.0
    cons.end_inv = 20.0
    cons.inj_cost = 0.2
    cons.rel_cost = 0.1
    cons.rmax = 1.
    cons.rmin = -1.

    model_params = ModelParams()
    model_params.sigma_1 = 0.5
    model_params.sigma_2 = 0.1
    model_params.alpha_1 = 8.
    model_params.alpha_2 = 1.
    model_params.rho = np.eye(2, dtype=float)

    P = 2.0 * np.pi * 10.
    delta_F = 4.
    Fc_mean = 20.
    grad = 2.

    fwd_array = Fc_mean + delta_F * np.sin(-0.5*np.pi + np.linspace(0., P, nTe))
    # fwd_array += 0.4 * delta_F * np.sin(0.2 + np.linspace(0., 2.0 * P, nTe))
    # fwd_array += grad * np.linspace(0., 1., nTe)

    fwd_curve = pd.Series(fwd_array, dates)

    ret = dict()
    ret['val_date'] = val_date
    ret['dates'] = dates
    ret['cons'] = cons
    ret['model_params'] = model_params
    ret['fwd_curve'] = fwd_curve
    ret['Fc_mean'] = Fc_mean
    ret['delta_F'] = delta_F

    return ret

def setting_2():

    val_date = pd.datetime(2015, 1, 1)
    start_date = pd.datetime(2015, 1, 1)
    nTe = 300
    dates = pd.date_range(start_date, periods=nTe)

    cons = Constraints()
    cons.max_inv = 50.
    cons.min_inv = 0.
    cons.start_inv = 20.0
    cons.end_inv = 20.0
    cons.inj_cost = 0.2
    cons.rel_cost = 0.1
    cons.rmax = 1.
    cons.rmin = -1.

    model_params = ModelParams()
    model_params.sigma_1 = 0.5
    model_params.sigma_2 = 0.1
    model_params.alpha_1 = 8.
    model_params.alpha_2 = 1.
    model_params.rho = np.eye(2, dtype=float)

    P = 2.0 * np.pi * 10.
    delta_F = 4.
    Fc_mean = 20.
    grad = 2.

    # fwd_array = Fc_mean + delta_F * np.sin(-np.pi + np.linspace(0., P, nTe))
    fwd_array = Fc_mean + delta_F * np.sin(-0.5*np.pi + np.linspace(0., P, nTe))
    fwd_array += 0.4 * delta_F * np.sin(0.2 + np.linspace(0., 2.0 * P, nTe))
    # fwd_array += grad * np.linspace(0., 1., nTe)

    fwd_curve = pd.Series(fwd_array, dates)

    ret = dict()
    ret['val_date'] = val_date
    ret['dates'] = dates
    ret['cons'] = cons
    ret['model_params'] = model_params
    ret['fwd_curve'] = fwd_curve
    ret['Fc_mean'] = Fc_mean
    ret['delta_F'] = delta_F

    return ret


def setting_3():

    val_date = pd.datetime(2015, 1, 1)
    start_date = pd.datetime(2015, 1, 1)
    nTe = 300
    dates = pd.date_range(start_date, periods=nTe)

    cons = Constraints()
    cons.max_inv = 50.
    cons.min_inv = 0.
    cons.start_inv = 0.0
    cons.end_inv = 0.0
    cons.inj_cost = 0.2
    cons.rel_cost = 0.1
    cons.rmax = 1.
    cons.rmin = -1.

    model_params = ModelParams()
    model_params.sigma_1 = 0.5
    model_params.sigma_2 = 0.1
    model_params.alpha_1 = 8.
    model_params.alpha_2 = 1.
    model_params.rho = np.eye(2, dtype=float)

    P = 2.0 * np.pi * 10.
    delta_F = 4.
    Fc_mean = 20.
    grad = 2.

    fwd_array = Fc_mean + delta_F * np.sin(-0.5*np.pi + np.linspace(0., P, nTe))
    fwd_array += 0.4 * delta_F * np.sin(0.2 + np.linspace(0., 2.0 * P, nTe))
    fwd_array += grad * np.linspace(0., 1., nTe)

    fwd_curve = pd.Series(fwd_array, dates)

    ret = dict()
    ret['val_date'] = val_date
    ret['dates'] = dates
    ret['cons'] = cons
    ret['model_params'] = model_params
    ret['fwd_curve'] = fwd_curve
    ret['Fc_mean'] = Fc_mean
    ret['delta_F'] = delta_F

    return ret

def get_setting(n):
    if n == 1:
        return setting_1()
    if n == 2:
        return setting_2()
    if n == 3:
        return setting_3()

    if 300 <= n < 400:
        ret = setting_3()
        ret['model_params'].alpha_1 = 0.2 * (n - 300.)
        return ret




def test_formula(nsims=None):

    # val_date, dates, cons, model_params, fwd_curve, Fc, delta_F = setting_1()
    # val_date, dates, cons, model_params, fwd_curve, Fc, delta_F = setting_2()
    val_date, dates, cons, model_params, fwd_curve, Fc, delta_F = setting_3()

    Te = calcTe(val_date, dates)
    Te_ext = np.append(Te, Te[-1] + 1./365.)
    nTe = Te.size
    fwd_array = fwd_curve.values


    # estimate mean forward curve and gradient upper bound

    F0_mean = fwd_array.mean()
    dt = 1./365.
    dFdt = np.gradient(fwd_array, dt)
    F_grad_rms = rms(dFdt)
    F_grad_max = abs(dFdt).max()


    results, TP = run_one_valuation_intrinsic(Te, fwd_array, cons)

    cashflows = results['cashflows'].spot + results['cashflows'].opex
    mean_cf = cashflows.mean(axis=1)
    value = mean_cf.sum()
    mean_actions = results.actions.mean(axis=1)
    print "intrinsic value = {}".format(value)

    levels = np.empty(nTe+1, dtype=float)
    levels[0] = cons.start_inv
    levels[1:] = cons.start_inv + np.cumsum(mean_actions)

    minor_sticks = []
    for t in range(nTe)[1:]:
        if mean_actions[t] != mean_actions[t-1]:
            minor_sticks.append(Te[t])

    triggers = calculate_trigger_prices2(TP['dVdq'], Te, fwd_array, mean_actions, cons)
    # triggers = calculate_trigger_prices(TP['dVdq'], Te, fwd_array, mean_actions, cons)

    log_debug(Te, 'Te')
    log_debug(Te*365., 'Te*365')
    log_debug(fwd_array, 'fwd_array')
    log_debug(triggers.trigger_prices_upper, 'trigger_prices_upper')
    log_debug(triggers.trigger_prices_lower, 'trigger_prices_lower')
    log_debug(triggers.tt_up*365., 'tt_up*365')
    log_debug(triggers.tt_down*365., 'tt_down*365')
    log_debug(triggers.fdot_up, 'fdot_up')
    log_debug(triggers.fdot_down, 'fdot_down')


    if nsims is not None and nsims > 0:
        stoch_value, stoch_value_error = run_valuation_stochastic(val_date,
                                                                  dates,
                                                                  fwd_curve,
                                                                  cons,
                                                                  model_params,
                                                                  nsims=nsims)
        print "stoch value = {:.4g} +- {:.4g}".format(stoch_value, stoch_value_error)
        print "time value = {:.4g} +- {:.4g}".format(stoch_value - value, stoch_value_error)

    plt.close('all')
    plt.figure()
    plt.plot(Te, fwd_array, 'o-', linewidth=1, markersize=2)
    plt.plot(Te_ext, levels)
    plt.plot(Te, triggers.trigger_prices, 'o-', linewidth=1, markersize=2)
    plt.plot(Te, triggers.trigger_prices_upper, '--', linewidth=1)
    plt.plot(Te, triggers.trigger_prices_lower, '--', linewidth=1)
    ax = plt.gca()
    # start, end = ax.get_xlim()
    ax.xaxis.set_ticks(minor_sticks, minor=True)
    plt.grid(True)
    ax.xaxis.grid(True, which='minor')
    # plt.show()

    # return

    #############################################################
    if Te_ext[0] > 0:
        Num_0_Ta = Te_ext[0] * 365.
        tenors_0_Ta = np.linspace(0., Te_ext[0], Num_0_Ta+1)
        tenors_0_Tb = np.append(tenors_0_Ta, Te_ext[1:])
    else:
        tenors_0_Tb = Te_ext

    integration_dt = np.diff(tenors_0_Tb)

    log_debug(Te_ext*365., 'tenors_Ta_Tb*365')
    log_debug(tenors_0_Tb*365., 'tenors_0_Tb*365')

    from mkl.interpolation import Interpolate1D
    triggers_matrix = np.hstack((triggers.trigger_prices_upper.reshape(-1, 1),
                                 triggers.trigger_prices_lower.reshape(-1, 1),
                                 fwd_array.reshape(-1, 1)))
    trigger_interpolator = Interpolate1D(Te, triggers_matrix, kind='linear')

    num_time_steps = tenors_0_Tb.size

    rmax = cons.rmax
    rmin = cons.rmin

    rmax *= 365
    rmin *= 365

    time_value1 = 0
    time_value2 = 0
    time_value3 = 0

    time_value_evolution = np.zeros(num_time_steps)

    I1_evolution1 = np.zeros(num_time_steps - 1)
    I2_evolution1 = np.zeros(num_time_steps - 1)
    mu_evolution1 = np.zeros(num_time_steps - 1)

    I1_evolution2 = np.zeros(num_time_steps - 1)
    I2_evolution2 = np.zeros(num_time_steps - 1)
    mu_evolution2 = np.zeros(num_time_steps - 1)


    for n in range(num_time_steps - 1):
        log_function_enter('time loop')
        log_debug('time step {} of {}'.format(n, num_time_steps - 1), std_out=False)
        t = tenors_0_Tb[n]
        tm = max(t, Te[0])
        Tb = tenors_0_Tb[-1]
        dt = 1./365.
        num_steps = int(np.round((Tb - tm) / dt))
        num_steps *= 5.
        tenors = np.linspace(tm, Tb, num_steps + 1)

        log_debug(t*365., 't*365')

        C_all = trigger_interpolator.evaluate(tenors).reshape(-1, 3).T
        C_u = C_all[0]
        C_d = C_all[1]
        F0 = C_all[2]

        params = dict()
        params['n'] = n
        params['t'] = t
        params['tenors'] = tenors
        params['model_params'] = model_params
        params['C_u'] = C_u
        params['C_d'] = C_d
        params['F0'] = F0
        params['rmax'] = rmax
        params['rmin'] = rmin
        params['F0_mean'] = F0_mean
        params['F_grad_max'] = F_grad_max

        add_params = dict()
        add_params['wmin_steps'] = 10.
        add_params['k_thr'] = 0.7
        add_params['correction_weight'] = 0.5

        mu_1, I1_1, I2_1 = calculate_mu_bar(params, add_params)
        mu_2, I1_2, I2_2 = calculate_mu_bar_2(t, model_params, rmax, rmin, triggers)

        time_value1 += mu_1 * integration_dt[n]
        time_value2 += mu_2 * integration_dt[n]
        time_value_evolution[n+1] = time_value1

        log_debug(mu_1, 'mu_1')
        log_debug(I1_1, 'I1_1')
        log_debug(I2_1, 'I2_1')
        log_debug(time_value1, 'time value 1')

        I1_evolution1[n] = I1_1
        I2_evolution1[n] = I2_1
        mu_evolution1[n] = mu_1

        I1_evolution2[n] = I1_2
        I2_evolution2[n] = I2_2
        mu_evolution2[n] = mu_2

        log_function_exit('time loop')

    log_debug(time_value1, 'final time_value')
    print "approximated time value1 = {:.4g}".format(time_value1)
    print "approximated time value2 = {:.4g}".format(time_value2)

    # plt.figure()
    # plt.plot(tenors_0_Tb, time_value_evolution)
    # plt.title('time value')
    # plt.grid(True)

    plt.figure()
    plt.plot(tenors_0_Tb[:-1], mu_evolution1, '--r', label='mu1')
    plt.plot(tenors_0_Tb[:-1], I1_evolution1, 'r', label='I1_evolution_1')
    plt.plot(tenors_0_Tb[:-1], I2_evolution1, 'r', label='I2_evolution_1')

    plt.plot(tenors_0_Tb[:-1], mu_evolution2, '--g', label='mu2')
    plt.plot(tenors_0_Tb[:-1], I1_evolution2, 'g', linewidth=2, label='I1_evolution_2')
    plt.plot(tenors_0_Tb[:-1], I2_evolution2, 'g', label='I2_evolution_2')

    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def run_optimisation():

    setups = [304, 310, 340]

    # get lsmc values
    true_values, errors = get_lsmc_values(setups, nsims=20000)
    log_debug(true_values, 'lsmc time values', std_out=True)

    results = []
    best_case = None
    for wmin_steps in [1., 5., 10.]:
        for k_thr in [0.6, 0.7, 0.8]:
            for correction_weight in [0.7, 0.8, 0.9, 1.]:

                start_time = log_function_enter('combination {}'.format(np.array([wmin_steps, k_thr, correction_weight])))
                log_debug([wmin_steps, k_thr, correction_weight], 'considering combination', std_out=True)

                cut_off_params = dict()
                cut_off_params['wmin_steps'] = wmin_steps
                cut_off_params['k_thr'] = k_thr
                cut_off_params['correction_weight'] = correction_weight


                # get approximation
                app_values = get_formula_values(cut_off_params, setups)

                x = abs(app_values - true_values) - errors
                x[x < 0.] = 0.
                rel_errors =  x / true_values
                case_error = rel_errors.max()
                log_debug(app_values, 'appoximation_values', std_out=True)
                log_debug(rel_errors, 'rel_errors', std_out=True)
                results.append([case_error, wmin_steps, k_thr, correction_weight])
                if best_case is None or case_error < best_case[0]:
                    best_case = [case_error, wmin_steps, k_thr, correction_weight]
                    log_debug('best case so far', std_out=True)

                log_function_exit('combination', start_time=start_time)

    log_debug(best_case, 'best_case', std_out=True)
    log_debug(results, 'all results', std_out=True)


def run_valuation_series():

    results = []

    for setup in range(300, 310):

        start_time = log_function_enter('valuation for setting {}'.format(setup))
        # get lsmc values
        true_values, errors = get_lsmc_values([setup], nsims=20000)
        log_debug(true_values[0], 'lsmc time value', std_out=True)


        cut_off_params = dict()
        cut_off_params['wmin_steps'] = 5.
        cut_off_params['k_thr'] = 0.75
        cut_off_params['correction_weight'] = 0.9


        # get approximation
        app_value = get_formula_values(cut_off_params, [setup])[0]

        x = max(abs(app_value - true_values[0]) - errors[0], 0.)
        rel_error =  2. * x / (true_values[0] + app_value)
        log_debug(app_value, 'appoximation_value', std_out=True)
        log_debug(rel_error, 'rel_error', std_out=True)
        results.append([true_values[0], app_value, rel_error])
        log_function_exit('valuation for setting {}'.format(setup), start_time)

    log_debug(results, 'results', std_out=True)


@log()
def get_lsmc_values(setups, nsims=5000):

    values = []
    errors = []
    for num_setting in setups:
        log_debug(num_setting, 'setting')

        ######### load setting

        ret = get_setting(num_setting)
        val_date = ret['val_date']
        dates = ret['dates']
        cons = ret['cons']
        model_params = ret['model_params']
        fwd_curve = ret['fwd_curve']
        Fc_mean = ret['Fc_mean']
        delta_F = ret['delta_F']


        Te = calcTe(val_date, dates)
        Te_ext = np.append(Te, Te[-1] + 1./365.)
        nTe = Te.size
        fwd_array = fwd_curve.values




        Te = calcTe(val_date, dates)
        fwd_array = fwd_curve.values

        ########## run instrinsic
        results, TP = run_one_valuation_intrinsic(Te, fwd_array, cons)

        cashflows = results['cashflows'].spot + results['cashflows'].opex
        mean_cf = cashflows.mean(axis=1)
        intr_value = mean_cf.sum()

        ########## run lsmc
        stoch_value, stoch_value_error = run_valuation_stochastic(val_date,
                                                                  dates,
                                                                  fwd_curve,
                                                                  cons,
                                                                  model_params,
                                                                  nsims=nsims)
        time_value = stoch_value - intr_value

        log_debug("stoch value = {:.4g} +- {:.4g}".format(stoch_value, stoch_value_error))
        log_debug("time value = {:.4g} +- {:.4g}".format(time_value, stoch_value_error))
        values.append(time_value)
        errors.append(stoch_value_error)
    return np.array(values), np.array(errors)

@log()
def get_formula_values(cut_off_params, setups):


    wmin_steps = cut_off_params['wmin_steps']
    k_thr = cut_off_params['k_thr']
    correction_weight = cut_off_params['correction_weight']


    values = []
    for num_setting in setups:

        start_time0 = log_function_enter('setting {}'.format(num_setting))

        ret = get_setting(num_setting)
        val_date = ret['val_date']
        dates = ret['dates']
        cons = ret['cons']
        model_params = ret['model_params']
        fwd_curve = ret['fwd_curve']
        Fc_mean = ret['Fc_mean']
        delta_F = ret['delta_F']


        Te = calcTe(val_date, dates)
        Te_ext = np.append(Te, Te[-1] + 1./365.)
        nTe = Te.size
        fwd_array = fwd_curve.values


        # estimate mean forward curve and gradient upper bound

        F0_mean = fwd_array.mean()
        dt = 1./365.
        dFdt = np.gradient(fwd_array, dt)
        F_grad_max = abs(dFdt).max()

        results, TP = run_one_valuation_intrinsic(Te, fwd_array, cons)
        mean_actions = results.actions.mean(axis=1)

        levels = np.empty(nTe+1, dtype=float)
        levels[0] = cons.start_inv
        levels[1:] = cons.start_inv + np.cumsum(mean_actions)

        minor_sticks = []
        for t in range(nTe)[1:]:
            if mean_actions[t] != mean_actions[t-1]:
                minor_sticks.append(Te[t])

        triggers = calculate_trigger_prices2(TP['dVdq'], Te, fwd_array, mean_actions, cons)

        #############################################################
        if Te_ext[0] > 0:
            Num_0_Ta = Te_ext[0] * 365.
            tenors_0_Ta = np.linspace(0., Te_ext[0], Num_0_Ta+1)
            tenors_0_Tb = np.append(tenors_0_Ta, Te_ext[1:])
        else:
            tenors_0_Tb = Te_ext

        integration_dt = np.diff(tenors_0_Tb)

        triggers_matrix = np.hstack((triggers.trigger_prices_upper.reshape(-1, 1),
                                     triggers.trigger_prices_lower.reshape(-1, 1),
                                     fwd_array.reshape(-1, 1)))
        trigger_interpolator = Interpolate1D(Te, triggers_matrix, kind='linear')

        num_time_steps = tenors_0_Tb.size

        rmax = cons.rmax
        rmin = cons.rmin

        rmax *= 365
        rmin *= 365

        time_value1 = 0

        time_value_evolution = np.zeros(num_time_steps)

        I1_evolution1 = np.zeros(num_time_steps - 1)
        I2_evolution1 = np.zeros(num_time_steps - 1)
        mu_evolution1 = np.zeros(num_time_steps - 1)

        start_time = log_function_enter('integration time loop')
        for n in range(num_time_steps - 1):

            # log_debug('time step {} of {}'.format(n, num_time_steps - 1), std_out=False)
            t = tenors_0_Tb[n]
            tm = max(t, Te[0])
            Tb = tenors_0_Tb[-1]
            dt = 1./365.
            num_steps = int(np.round((Tb - tm) / dt))
            num_steps *= 5.
            tenors = np.linspace(tm, Tb, num_steps + 1)

            # log_debug(t*365., 't*365')

            C_all = trigger_interpolator.evaluate(tenors).reshape(-1, 3).T
            C_u = C_all[0]
            C_d = C_all[1]
            F0 = C_all[2]


            params = dict()
            params['n'] = n
            params['t'] = t
            params['tenors'] = tenors
            params['model_params'] = model_params
            params['C_u'] = C_u
            params['C_d'] = C_d
            params['F0'] = F0
            params['rmax'] = rmax
            params['rmin'] = rmin
            params['F0_mean'] = F0_mean
            params['F_grad_max'] = F_grad_max

            add_params = dict()
            add_params['wmin_steps'] = wmin_steps
            add_params['k_thr'] = k_thr
            add_params['correction_weight'] = correction_weight

            mu_1, I1_1, I2_1 = calculate_mu_bar(params, add_params)

            time_value1 += mu_1 * integration_dt[n]
            time_value_evolution[n+1] = time_value1

            # log_debug(mu_1, 'mu_1')
            # log_debug(I1_1, 'I1_1')
            # log_debug(I2_1, 'I2_1')
            # log_debug(time_value1, 'time value 1')

            I1_evolution1[n] = I1_1
            I2_evolution1[n] = I2_1
            mu_evolution1[n] = mu_1

        log_function_exit('integration time loop', start_time=start_time)
        log_debug(time_value1, 'final time_value')
        # print "approximated time value1 = {:.4g}".format(time_value1)
        values.append(time_value1)
        log_function_exit('setting {}'.format(num_setting), start_time=start_time0)
    return np.array(values)



def plot_surface(tenors, Z, title_str=''):
    X, Y = np.meshgrid(tenors, tenors)
    X = X.T
    Y = Y.T

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)

    # surf = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.contour(X, Y, Z, zdir='x', offset=0.15, cmap=cm.coolwarm)
    ax.contour(X, Y, Z, zdir='y', offset=0.32, cmap=cm.coolwarm)
    ax.contour(X, Y, Z, zdir='z', offset=-0.1, cmap=cm.coolwarm)

    # ax.set_xlim3d(0.15, 0.3)
    # ax.set_ylim3d(0.2, 0.32)
    # ax.set_zlim3d(-0.1, 0.3)
    plt.title(title_str)
    # plt.show()





if __name__ == "__main__":
    open_logger("test_trigger_price9.log")
    # test_trigger_prices()
    # test_formula(nsims=5000)

    run_optimisation()
    # run_valuation_series()

    # val_date, dates, cons, model_params, fwd_curve, Fc_mean, delta_F = setting_3()
    # plt.plot(fwd_curve)
    # plt.grid(True)
    # plt.show()





