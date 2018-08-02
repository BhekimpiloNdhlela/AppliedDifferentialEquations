#usr/bin/env/python
'''
Author  : Bhekimpilo Ndhlela
Author  : 18998712
module  : TW244 Applied Mathematics
task    : Assignment 01 problem2
since   : Saturday-28-07-2018
'''

from numpy import (exp, linspace, array, abs, zeros, shape)
import matplotlib.pyplot as plt
from sys import exit

def eulers_method(f, I, y0=1.0,(a,b)=(0.0,4.0), h=1.0):
    X = array([0.0, 1.0, 2.0, 3.0, 4.0])
    W = array([y0, 0.0, 0.0, 0.0, 0.0])

    for i in xrange(int(a), int(b)):
        W[i+1] = W[i] + h * f(X[i], W[i])
    absolute_error(W, X, 'Euler\'s Method')

def modified_eulers_method(f, I, y0=1.0,(a,b)=(0.0,4.0), h=1.0):
    X = array([0.0, 1.0, 2.0, 3.0, 4.0])
    W = array([y0, 0.0, 0.0, 0.0, 0.0])

    for i in xrange(int(a), int(b)):
        temp = W[i] + h*f(X[i], W[i])
        W[i+1] = W[i] + (h/2.0)*(f(X[i], W[i]) + f(X[i+1], temp))
    absolute_error(W, X, 'Improved Euler\'s Method')

def absolute_error(W, X, label, debug=True):
    Y = array([ I(n) for n in xrange(len(W))])
    abs_err = array([abs(ya - yc) for ya, yc in zip(Y, W)])
    if debug is True:
        print(label)
        for i , (err, w) in enumerate(zip(abs_err, W)):
            print 'x = ', i, '\t\tw = {:.10f} \ty = {:.10f}\terr = {:.10f}'.format(w, Y[i], abs_err[i])
    plot_comparison_func(Y, W, X, label)

def plot_comparison_func(Y, W, x, lbl):
    plt.title('Analytical vs. ' + lbl)
    plt.xlabel('x')
    plt.ylabel('y = f(x)')
    plt.plot(x, Y, '-k', linewidth=2, label='Analytical Solution')
    plt.plot(x, W, '--r', linewidth=2, label='Numerical Solution')
    plt.legend(bbox_to_anchor=(.4, .4))
    plt.show()

def plot_analytical_solution():
    X = linspace(0, 4, 1000)
    y = array([ exp(1 - exp(-x)) for x in X])
    plt.title('Plot of function f(x) = exp(1 - exp(-x))')
    plt.xlabel('x = linspace(0, 4, 1000)')
    plt.ylabel('f(x) = exp(1 - exp(-x))')
    plt.plot(X, y, '-k', linewidth=4)
    plt.show()

if __name__ == '__main__':
    f = lambda x, y: exp(-x) * y    # f(x, y) = y DE
    I = lambda x: exp(1 - exp(-x))  # Exact Solution
    plot_analytical_solution()
    eulers_method(f, I)
    modified_eulers_method(f, I)
else:
    exit('USAGE: python problem2.py')
