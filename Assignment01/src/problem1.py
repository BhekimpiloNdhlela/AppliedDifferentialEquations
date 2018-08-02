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

def plot_solution_func_problem_b():
    X = linspace(0, 2.5, 100)
    y = array([ p**3 - 4*p**2 +4*p for p in X])
    plt.title('Plot of function f(x) = exp(1 - exp(-x))')
    plt.xlabel('x = linspace(0, 4, 1000)')
    plt.ylabel('f(x) = exp(1 - exp(-x))')
    plt.plot(X, y, '-k', linewidth=4)
    plt.show()
'''
def plot_solution_func_problem_b():
    X = linspace(0, 2.1, 100)
    y = array([ p*(p-2)**3 * (3*p - 2) for p in X])
    plt.title('Plot of function f(x) = exp(1 - exp(-x))')
    plt.xlabel('x = linspace(0, 4, 1000)')
    plt.ylabel('f(x) = exp(1 - exp(-x))')
    plt.plot(X, y, '-k', linewidth=4)
    plt.show()
'''
if __name__ == '__main__':
    plot_solution_func_problem_b()
else:
    exit('USAGE: python problem2.py')
