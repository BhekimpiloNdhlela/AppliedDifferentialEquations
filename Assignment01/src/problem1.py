#usr/bin/env/python
'''
Author  : Bhekimpilo Ndhlela
Author  : 18998712
module  : TW244 Applied Mathematics
task    : Assignment 01 problem2
since   : Saturday-28-07-2018
'''

from numpy import (linspace, array, zeros)
import matplotlib.pyplot as plt
from sys import exit

def plot_functions():
    
    X = linspace(0, 2.5, 100)
    y = array([ p**3 - 4*(p**2) +4*p for p in X])
    plt.subplot(211)
    plt.title('Plot of function f\'(x)')
    plt.xlabel('x = linspace(0, 2.5, 1000)')
    plt.ylabel('f(x)')
    plt.plot(X, y, '-r', linewidth=4)
    
    plt.subplot(212)
    X = linspace(0, 2.5, 100)
    y = array([ p*(p-2)**3 * (3*p - 2) for p in X])
    plt.title('Plot of function f\'\'(x)')
    plt.xlabel('x = linspace(0, 2.5, 1000)')
    plt.ylabel('f(x)')
    plt.plot(X, zeros(len(X)), '-k', linewidth=2)
    plt.plot(X, y, '-r', linewidth=4)
    plt.show()

plot_functions()
