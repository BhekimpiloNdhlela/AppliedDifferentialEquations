#!/usr/bin/env python

"""
Author  : Bhekimpilo Ndhlela
Author  : 18998712
module  : TW244 Applied Mathematics
task    : src code for Assignment04
"""
from scipy.special import lambertw
from scipy.optimize import fsolve
from scipy.linalg import eig, expm
import matplotlib.pyplot as plt
import numpy as np

def plot_question3(x, y, point=None, point_plot=False):
    plt.title("IVP solution Plot:")
    plt.plot(x, y[:,0], 'k-', lw=4, label='x1')
    plt.plot(x, y[:,1], 'r-', lw=4, label='x2')
    plt.ylabel('y')
    if point_plot != False:
        plt.plot(10, point[0], 'ro', lw=4, label='x1(10)')
        plt.plot(10, point[1], 'ko', lw=4, label='x2(10)')
        plt.xlabel('0 <= t <= 10')
        plt.xlim(0, 10)
    else:
        plt.xlabel('0 <= t <= 2')
        plt.xlim(0, 2)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def f(R):
    sind = lambda theta_radians: np.degrees(np.sin(theta_radians))
    cosd = lambda theta_radians: np.degrees(np.cos(theta_radians))
    g, r, theta, v0 = 9.81, 0.14, 45.0, 44.7 # constants
    print (g/r + v0*sind(theta))*(R/(v0*cosd(theta))) + (g/r**2)*np.log(1-(r*R)/(v0*cosd(theta)))
    return (g/r + v0*sind(theta))*(R/(v0*cosd(theta))) + (g/r**2)*np.log(1-(r*R)/(v0*cosd(theta)))

if __name__ == '__main__':
    # question 1
    #a.)
    z = fsolve(lambda z: z*np.exp(z) - 1.0, 0.0)
    #print z
    #b.)    i.
    w = lambertw(1).real
    #print w
    #b.)    ii.
    wn = w *np.exp(w)
    #print wn

    # question 2
    print '{:.17f}'.format(1.7356e-14 )
    R = fsolve(f, )
    print 'R = ', R

    # question 3
    x0 = np.array([2, -0.5])
    A, time = np.array([[-0.50, 3.0], [0.75, -2.0]]), np.linspace(0, 2, 1000)
    eigvals, eigvects = eig(A) #eigenvaluse and eigen vectors of: A
    eigvals = eigvals.real #take the real part of the eigvalues
    c = np.linalg.solve(eigvects, x0)   #z = inverse(eigenvectors)b

    x1 = lambda t: c[0]*np.exp(eigvals[0]*t)*eigvects[0][0] + c[1]*np.exp(eigvals[1]*t)*eigvects[0][1]
    x2 = lambda t: c[0]*np.exp(eigvals[0]*t)*eigvects[1][0] + c[1]*np.exp(eigvals[1]*t)*eigvects[1][1]

    x = np.array([(x1(t), x2(t)) for t in time])
    #plot_question3(time, x, point=None)

    eA = np.dot(expm(10 *A), x0) # get the matrix exponintiation and multiply by x(0)
    time = np.linspace(0, 10, 1000)
    x = np.array([(x1(t), x2(t)) for t in time])
    #plot_question3(time, x, point=eA, point_plot=True)

    #question 4

else:
    import sys
    sys.exit('USAGE: python assignment04.py')
