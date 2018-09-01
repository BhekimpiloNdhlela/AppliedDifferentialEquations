#usr/bin/env/python
'''
Author  : Bhekimpilo Ndhlela
Author  : 18998712
module  : TW244 Applied Mathematics
task    : Assignment 03 problem3
since   : Wednesday-29-08-2018
'''

from numpy import exp, arange, log, array
import matplotlib.pyplot as plt
from scipy.special import expi
import scipy.integrate

def plot_graphs_Q3(x, y, y1=[]):
    plt.xlabel('x')
    plt.ylabel('f(x) = y(x) = x')
    if y1 == []:
        plt.plot(x, y[:,0], '-k', linewidth=4)
        plt.title('Numerical Solution of: (x, f(x))')
    else:
        plt.title('Numerical and Analytical Solution of: (x, f(x))')
        plt.plot(x, y[:,0], '-k', linewidth=4, label='Numerical Solution')
        plt.plot(x, y1, '--r', linewidth=4, label='Analytical Solution')
        plt.legend(loc='best')
    plt.grid(True, linewidth=3)
    plt.xlim([1, 4])
    plt.show()

if __name__ == '__main__':
    # question a.)
    X, x0 = arange(1, 4.01, 1./1000), [0, 1]
    F_num = lambda x, t: (x[1], log(t) - 2*x[1] - x[0])
    Y0 = scipy.integrate.odeint(F_num, x0, X)#[:,0]
    plot_graphs_Q3(X, Y0)

    # question b.)
    F_ana = lambda x: exp(-x)*((-expi(-1)+expi(-x))*(1+x)+(x-2)*exp(1))+1+log(x)
    Y1 = array([F_ana(x) for x in X])
    plot_graphs_Q3(X, Y0, y1=Y1)

else:
    exit('USAGE: python problem3.py')
