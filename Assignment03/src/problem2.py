#usr/bin/env/python
'''
Author  : Bhekimpilo Ndhlela
Author  : 18998712
module  : TW244 Applied Mathematics
task    : Assignment 03 problem2
since   : Wednesday-29-08-2018
'''
import matplotlib.pyplot as plt
import scipy.integrate
import numpy as np

def plot_graphs_Q2(odeint_sol, t):
    plt.title('S, I, and R against t')
    plt.xlabel('t = time')
    plt.ylabel('S, I, R')
    plt.plot(t, odeint_sol[:,0], '-r', linewidth=4, label='number Susceptible')
    plt.plot(t, odeint_sol[:,1], '-k', linewidth=4, label='number Infectious')
    plt.plot(t, odeint_sol[:,2], '-b', linewidth=4, label='number Recovered')
    plt.legend(loc='best')
    plt.grid(True, linewidth=3)
    plt.xlim([0, t[-1]])
    plt.show()

if __name__ == '__main__':
    # problem b.)
    B, G, T, x0 = .00083, .05, np.arange(0,28,1./100.), [999., 1., 0.]
    f = lambda x, t: (-B*x[1]*x[0], B*x[1]*x[0] - G*x[1], G*x[1])
    odeint_sol = scipy.integrate.odeint(f, x0, T)
    plot_graphs_Q2(odeint_sol, T)

    # problem c.)
    # i.]
    max_I = max(odeint_sol[:,1])
    # ii.]
    I_28days = odeint_sol[:,1][-1]
    print 'Maximum Number of Infected Students  : ', int(round(max_I))
    print 'Infected Students in 28 Days         : ', int(round(I_28days))

    # problem d.)
    '''
    by trial and error: Gamma = 0.6
    '''
    G, T = 0.6, np.arange(0, 56, 1./100.)
    odeint_sol = scipy.integrate.odeint(f, x0, T)
    plot_graphs_Q2(odeint_sol, T)

    # problem e.)
    #The model would be: dS/dt = -b*IS + d*I dI/dt = b*IS - (g+d)*I dR/dt = g*I

else:
    exit('USAGE: python problem2.py')
