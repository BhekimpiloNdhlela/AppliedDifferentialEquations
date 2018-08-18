#!/usr/bin/python
"""
author : Bhekimpilo Ndhlela
author : 18998712
module : TW244 Applied Mathematics
task   : Assignment 02 2018
since  : Monday - 13 - August - 2018
"""
import matplotlib.pyplot as plt
from sys import exit
import numpy as np

def plot4q1a(Y, P, Q):
    plt.subplot(121)
    plt.title('Scatter Plot of: (t+1790, P(t))')
    plt.xlabel('t + 1790')
    plt.ylabel('P(t)')
    plt.plot(Y, P, 'or', linewidth=4)

    plt.subplot(122)
    plt.title('Scatter Plot of: (P(t), Q(t))')
    plt.xlabel('P(t)')
    plt.ylabel('Q(t)')
    plt.plot(P[:len(Q)], Q, 'ob', linewidth=4)
    plt.show()

def plot4q1b(x, y, a, b):
    plt.title('Scatter-Plot & Line-of-Best-Fit for: (P(t), Q(t))')
    plt.xlabel('P(t)')
    plt.ylabel('Q(t)')
    plt.plot(x, y, 'ob', linewidth=4)
    best_fit = np.array([a + b*x[i] for i in xrange(len(y))])
    plt.plot(x, best_fit, '-c', linewidth=4)
    plt.show()

def plot4q1c(Y, P, x, y):
    plt.title('Scatter Plot & Curve-of-Best-Fit for: (t+1790, P(t))')
    plt.xlabel('t + 1790')
    plt.ylabel('P(t)')
    plt.plot(Y, P, 'or', linewidth=4)
    plt.plot(x, y, '-c', linewidth=4)
    plt.xlim([1790, 2018])
    plt.show()

def get_coef(x, y, deg):
    coef = np.polyfit(x, y, deg)
    return coef[1], coef[0]

if __name__ == '__main__':
    Y = np.arange(1790, 1960, 10)
    P = np.array([3.929,  5.308,  7.240,  9.368,  12.866,\
                  17.069, 23.192, 31.433, 38.558, 50.156,\
                  62.948, 75.996, 91.972, 105.711,122.775,\
                  131.669, 150.697])
    Q = lambda t: 1.0/10.0 * (year_pop[t+10]/year_pop[t] - 1)
    year_pop = {year: population for year, population in zip(Y, P)}
    Qt = np.array([Q(Y[i]) for i in xrange(len(Y) - 1)])
    plot4q1a(Y, P, Qt)

    # question 1 b.)
    a, b = get_coef(P[:len(Qt)], Qt, 1)
    plot4q1b(P[:len(Qt)], Qt, a, b)
    print 'a = ',a
    print 'b = ', b

    # question 1c
    P0 = 3.929
    Ps = lambda t: (a*P0)/(b*P0+(a-b*P0)*np.exp(-a*t))
    X = np.linspace(1790, 2018, 1000)
    px = np.array([Ps(x) for x in X])
    plot4q1c(Y, P, X, px)


else:
    exit('USAGE: python problem01.py')
