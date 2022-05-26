import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import interpolate

# Opg 2

def utility(c,par):
    return c**(1-par.rho)/(1-par.rho)

def bequest(m,c,par):
    return par.nu*(m-c+par.kappa)**(1-par.rho)/(1-par.rho)

def v_last_period(c,m,par):
    return utility(c,par) + bequest(m,c,par)


def v(c,m,par,v_plus_interp):
    
    # a. expected value
    v_plus = 0.0
    for p,y in [(par.p,par.ybar+par.gamma*par.s+par.Delta),((1.0-par.p,par.ybar+par.gamma*par.s-par.Delta))]:
        
        # i. next period cash-on-hand
        a = m-c-par.tau*par.s
        m_plus = (1+par.r)*a + y
        
        # ii. next-period values
        v_plus_now = v_plus_interp([m_plus])[0]
        
        # iii. probability weighted sum
        v_plus += p*v_plus_now
    
    # b. total value
    return utility(c,par) + par.beta*v_plus


def solve_last_period(par):

    # a. allocate
    m_grid = np.linspace(par.m_min,par.m_max,par.Nm)
    v_func = np.empty(par.Nm)
    c_func = np.empty(par.Nm)

    # b. solve
    for i,m in enumerate(m_grid):

        # i. objective
        obj = lambda x: -v_last_period(x[0],m,par)

        # ii. optimizer
        x0 = m/2 # initial value
        result = minimize(obj,[x0],method='L-BFGS-B',bounds=((1e-8,m),))

        # iii. save
        v_func[i] = -result.fun
        c_func[i] = result.x
        
    return m_grid,v_func,c_func

def solve_single_period(par,v_plus_interp):

    # a. allocate
    m_grid = np.linspace(par.m_min,par.m_max,par.Nm)
    v_func = np.empty(par.Nm)
    c_func = np.empty(par.Nm)
    
    # b. solve
    for i,m in enumerate(m_grid):
        
        # i. objective
        obj = lambda x: -v(x[0],m,par,v_plus_interp)
        
        # ii. solve
        x0 = m/2 # initial guess
        result = minimize(obj,[x0],method='L-BFGS-B',bounds=((1e-8,m),))
        
        # iv. save
        v_func[i] = -result.fun
        c_func[i] = result.x[0]
     
    return m_grid,v_func,c_func


def solve(par):
    
    # a. solve period 2
    m2_grid,v2_func,c2_func = solve_last_period(par)
    
    # b. construct interpolator
    v2_func_interp = interpolate.RegularGridInterpolator([m2_grid], v2_func,
        bounds_error=False,fill_value=None)
    
    # b. solve period 1
    m1_grid,v1_func,c1_func = solve_single_period(par,v2_func_interp)
    
    return m1_grid,c1_func,m2_grid,c2_func,v1_func,v2_func