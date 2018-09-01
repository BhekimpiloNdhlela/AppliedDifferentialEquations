
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
from math import log
import numpy as np

def plot4q1a(Y, P, Q):
    plt.subplot(121)
    plt.title('Scatter Plot of: (t+1790, P(t))')
    plt.xlabel('t + 1790')
    plt.ylabel('P(t)')
    plt.plot(Y, P, 'og', linewidth=4)
    plt.grid(True, linewidth=3)

    plt.subplot(122)
    plt.title('Scatter Plot of: (P(t), Q(t))')
    plt.xlabel('P(t)')
    plt.ylabel('Q(t)')
    plt.plot(P[:len(Q)], Q, 'og', linewidth=4)
    plt.grid(True, linewidth=3)
    plt.show()

def plot4q1b(x, y, a, b):
    plt.title('Scatter-Plot & Q = a - bP: (P(t), Q(t))')
    plt.xlabel('P(t)')
    plt.ylabel('Q(t)')
    plt.plot(x, y, 'og', linewidth=4)
    best_fit = np.array([a - b*x[i] for i in xrange(len(y))])
    plt.plot(x, best_fit, '-k', linewidth=4)
    plt.grid(True, linewidth=3)
    plt.show()

def plot4q1c(Y, P, x, y):
    plt.title('Logistic Model')
    plt.xlabel('t + 1790')
    plt.ylabel('P(t)')
    plt.plot(Y, P, 'og', linewidth=4)
    plt.plot(x, y, '-k', linewidth=4)
    plt.xlim([1790, 2020])
    plt.grid(True, linewidth=3)
    plt.show()

def plot4q2b(x1, y1, x2, a1, b1, a2, b2):
    plt.title('Q = a - bP vs. Q ~ a - blnP')
    plt.xlabel('P(t)')
    plt.ylabel('Q(t)')
    plt.plot(x1, y1, 'og', linewidth=4)
    best_fit1 = np.array([a1 - b1*x1[i] for i in xrange(len(y1))])
    best_fit2 = np.array([a2 - b2*x2[i] for i in xrange(len(y1))])
    plt.plot(x1, best_fit1, '-k', linewidth=4, label='Q = a - bP')
    plt.plot(x1, best_fit2, '-b', linewidth=4, label='Q ~ a - blnP')
    plt.legend(loc=0)
    plt.grid(True, linewidth=3)
    plt.show()

def plot4q2c(Y, P, x, y1, y2):
    plt.title('Logistic vs Gompertz Model')
    plt.xlabel('t + 1790')
    plt.ylabel('P(t)')
    plt.plot(Y, P, 'og', linewidth=4)
    plt.plot(x, y1, '-k', linewidth=4, label='Logistic Model')
    plt.plot(x, y2, '-b', linewidth=4, label='Gompertz Model')
    plt.xlim([1790, 2020])
    plt.legend(loc=0)
    plt.grid(True, linewidth=3)
    plt.show()

def get_coef(x, y, deg):
    c = np.polyfit(x, y, deg)
    return c[1], -1*c[0] if (c[0] < 0) else c[0]

if __name__ == '__main__':
    Y = np.arange(1790, 1960, 10)
    P = np.array([3.929,  5.308,   7.240,   9.368,   12.866, 17.069,\
                  23.192, 31.433,  38.558,  50.156,  62.948, 75.996,\
                  91.972, 105.711, 122.775, 131.669, 150.697])
    year_pop = {year: population for year, population in zip(Y, P)}
    # -------------------------------------------------------------------------
    # question 1a.)
    # -------------------------------------------------------------------------
    Q = lambda t: 1.0/10.0 * (year_pop[t+10]/year_pop[t] - 1)
    Qt = np.array([Q(Y[i]) for i in xrange(len(Y) - 1)])
    plot4q1a(Y, P, Qt)

    # -------------------------------------------------------------------------
    # question 1b.)
    # -------------------------------------------------------------------------
    a, b = get_coef(P[:len(Qt)], Qt, 1)
    plot4q1b(P[:len(Qt)], Qt, a, b)
    
    # -------------------------------------------------------------------------
    # question 1c.)
    # -------------------------------------------------------------------------
    T = np.arange(0, 240, 10)
    Pt = lambda t: (a*P[0])/(b*P[0]+(a-b*P[0])*np.exp(-a*t))
    p1 = np.array([Pt(t) for t in T])
    plot4q1c(Y, P, T + 1790, p1)

    # -------------------------------------------------------------------------
    # problem 2b.)
    # -------------------------------------------------------------------------
    P2 = np.log(P)
    a2, b2 = get_coef(P2[:len(Qt)], Qt, 1)
    plot4q2b(P[:len(Qt)], Qt, P2[:len(Qt)], a, b, a2, b2)

    # -------------------------------------------------------------------------
    # problem 2c.)
    # -------------------------------------------------------------------------
    c = np.log(3.929)-(a2/b2)
    Pt = lambda t : np.exp((a2/b2)+c*np.exp(-b2*t))
    p2 = np.array([Pt(t) for t in T])
    plot4q2c(Y, P, T + 1790, p1, p2)

else:
    exit('USAGE: python Assignment02.py')
