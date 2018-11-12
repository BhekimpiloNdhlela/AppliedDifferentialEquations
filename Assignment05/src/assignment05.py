#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author  : Bhekimpilo Ndhlela
Author  : 18998712
module  : TW244 Applied Mathematics
task    : src code for Assignment04
"""
import matplotlib.pyplot as plt
import scipy.integrate
import numpy as np

def plot_question1(t, x, sp1lim, sp2lim):
    plt.subplot(121)
    plt.title('Plot of: (t, x(t)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(t, x[:,0], 'k-', linewidth=4)
    plt.grid(True, linewidth=3)
    plt.xlim(sp1lim[0], sp1lim[1])
    plt.subplot(122)
    plt.title('Plot of: (x(t), x\'(t))')
    plt.xlabel('x\'(t)')
    plt.ylabel('x(t))')
    plt.plot(x[:,0], x[:,1], 'b-', linewidth=4)
    plt.xlim(sp2lim[0], sp2lim[1])
    plt.grid(True, linewidth=3)
    plt.show()

def plot_numericalVSanalytical_solution(t, x0, x1):
    plt.title('Analytical vs. Numerical Solutioin')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(t, x0, 'k-', linewidth=4, label='Numeric Solution')
    plt.plot(t, x1, 'r--', linewidth=4, label='Analytic Solution')
    plt.legend(loc='best')
    plt.grid(True, linewidth=3)
    plt.ylim(-1.1, 1.1)
    plt.xlim(0, 6*np.pi)
    plt.show()

def plot_envelop_and_pumped_mass_sys(t, x, e, negative_envelop=True):
    plt.title('System and Envelop')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(t, x, 'k-', linewidth=2, label='Solution')
    plt.plot(t, e, 'r--', linewidth=2, label='Envelop')
    if negative_envelop is True: plt.plot(t, -e, 'r--', linewidth=2)
    plt.xlim(t[0], t[-1])
    plt.grid(True, linewidth=3)
    plt.legend(loc='best')
    plt.show()

def plot_3b_vs_3c(t, xb, xc):
    plt.title('Comparison of: part (c) and part (b) Plots')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.plot(t, xb, 'k-', linewidth=3, label='part (b.) plot')
    plt.plot(t, xc, 'r--', linewidth=3, label='part (c.) plot')
    plt.grid(True, linewidth=3)
    plt.legend(loc='best')
    plt.ylim(-7, 7)
    plt.xlim(0, 25)
    plt.show()

def g_num(x, t):
    '''
    numerical solution
    '''
    if t < 2*np.pi:                return (x[1], -16*x[0])
    elif 2*np.pi <= t <= 4*np.pi:  return (x[1], -16*x[0] + np.sin(4*t))
    else:                          return (x[1], -16*x[0])

def g_ana(t):
    '''
    analytical solution
    '''
    if t < 2*np.pi:                return np.cos(4.0*t)
    elif 2*np.pi <= t <= 4*np.pi:  return np.cos(4.0*t)+(1.0/32.0)*(np.sin(4.0*t)+(8*np.pi-4.0*t)*np.cos(4.0*t))
    else:                          return np.cos(4.0*t)-(np.pi*np.cos(4.0*t))/4.0

if __name__ == '__main__':
    ''' question 1 '''
    # 1a.)
    t, x0 = np.linspace(0, 10, 1000), np.array([1.0, 0.0])
    f = lambda x, t: (x[1], -4*x[0] - x[1])
    sol = scipy.integrate.odeint(f, x0, t)
    plot_question1(t, sol, (0, 10),(-0.5, 1))

    # 1b.)
    f = lambda x, t: (x[1], -4*x[0] - x[0]**3)
    sol = scipy.integrate.odeint(f, x0, t)
    plot_question1(t, sol, (0, 10),(-1.1, 1.1))

    # 1c.) (Note that (x')**2 won’t work.|x'|x' incorporates the direction of motion.)
    f = lambda x, t: (x[1], -4*x[0] - abs(x[1]) * x[1])
    sol = scipy.integrate.odeint(f, x0, t)
    plot_question1(t, sol, (0, 10),(-0.5, 1.1))

    ''' question 2 '''
    # 2a.)
    t = np.linspace(0, 6*np.pi, 1000)
    sol = scipy.integrate.odeint(g_num, x0, t)
    plot_question1(t, sol, (0, t[-1]),(-1.1, 1.1))
    # 2c.)
    x_analytical = np.array([g_ana(i) for i in t])
    plot_numericalVSanalytical_solution(t, sol[:,0], x_analytical)

    ''' question 3 '''
    # 3a.)
    t, x0 = np.linspace(0, 250, 2000), np.array([0.0, 0.0])
    f = lambda x, t: (x[1], -4*x[0] + np.cos((21.0/10.0)*t))
    sol = scipy.integrate.odeint(f, x0, t)
    # also find the amplitude form for the 3a.) function to plot the envelop
    E = 7./3 - 4./3 -1 # epsilon e << 1
    w, r = 2.0, 21.0/10.0
    envelop = lambda t: (2.0/(w**2 -r**2)) * np.sin(0.5*(w-r)*t)# * np.sin(0.5*(w+r)*t)
    e = np.array([envelop(i) for i in t])
    plot_envelop_and_pumped_mass_sys(t, sol[:,0], e)

    # 3b.)
    f = lambda x, t: (x[1], -4*x[0]+np.cos(2.0*t))
    solb = scipy.integrate.odeint(f, x0, t)
    # also find the amplitude form for the 3b.) function to plot the envelop
    envelop = lambda t: np.sin(E*t)/(4*E)
    e = np.array([envelop(i) for i in t])
    plot_envelop_and_pumped_mass_sys(t, solb[:,0], e)

    # 3c.)
    f = lambda x, t: (x[1], -4*x[0] -(1.0*x[1])/10.0 + np.cos(2.0*t))
    solc = scipy.integrate.odeint(f, x0, t)
    plot_3b_vs_3c(t, solb[:,0], solc[:,0])

    ''' question 4 '''
    T, x0 = np.linspace(0.0, 25.0, 1000), np.array([0.1, 0])
    f = lambda x, t: (x[1], -4*x[0]*(1.0 + np.sin(2.0*t)))
    sol = scipy.integrate.odeint(f, x0, T)
    e = np.array([0.1*np.exp((2.0*t)/11.0) for t in T])
    plot_envelop_and_pumped_mass_sys(T, sol[:,0], e, negative_envelop=False)

else:
    import sys
    sys.exit('USAGE: python assignment05.py')
