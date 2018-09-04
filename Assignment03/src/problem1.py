#usr/bin/env/python
'''
Author  : Bhekimpilo Ndhlela
Author  : 18998712
module  : TW244 Applied Mathematics
task    : Assignment 03 problem1
since   : Wednesday-29-08-2018
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
def eulers_method(fx, fy, T, x0, y0, h):
    x, y = np.zeros(len(T), dtype=float), np.zeros(len(T), dtype=float)
    x[0], y[0] = x0, y0

    for i in range(1, len(T)):
       x[i] = x[i-1] + h*fx(T[i], x[i-1], y[i-1])
       y[i] = y[i-1] + h*fy(T[i], x[i-1], y[i-1])
    return x, y

def plot_graphs(x, y, t, odeint_sol=None):
    plt.xlabel('t = time')
    plt.ylabel('Prey vs. Predetor')
    plt.plot(t, x, '-k', linewidth=4, label='X = Predetor Euler')
    plt.plot(t, y, '-r', linewidth=4, label='Y = Prey Euler')
    if odeint_sol is not None:
        plt.title('Euler Method vs. ODE45 using Prey and Predetor model')
        plt.plot(t, odeint_sol[:,0], '-m', linewidth=4, label='X = Predetor ODE_INT')
        plt.plot(t, odeint_sol[:,1], '-b', linewidth=4, label='Y = Prey ODE_INT')
    else:
        plt.title('Euler Method: Prey and Predetor model')
    plt.legend(loc='best')
    plt.grid(True, linewidth=3)
    plt.show()

def plot_graph_Q1e(odeint_sol, t):
    plt.xlabel('t = time = linspace(0, 20000, 1000)')
    plt.ylabel('Prey vs. Predetor')
    plt.title('Prey vs. Predetor model (10000 as limiting Capacity for the Prey)')
    plt.plot(t, odeint_sol[:,0], '-r', linewidth=4, label='X = Predetor')
    plt.plot(t, odeint_sol[:,1], '-b', linewidth=4, label='Y = Prey')
    plt.legend(loc='best')
    plt.grid(True, linewidth=3)
    plt.show()

if __name__ == '__main__':
    # question 1a.)
    fx = lambda t, x, y: -3*x + 3*x*y
    fy = lambda t, x, y: y - 2*x*y
    T, h, x0, y0 = np.arange(0, 10, 1.0/100.0), 1.0/100.0, 0.3, 1.0
    X, Y = eulers_method(fx, fy, T, x0, y0, h)
    #plot_graphs(X, Y, T)

    # question 1b.)
    '''
    (b) Using the same function f as in part (a), solve the same system using
    ode45 . Plot the solution (using a continuous line) on the same Figure as
    above using hold on . Show the plot and the code you used to call ode45.
    Compare with the plot from (a). Which do you think is more accurate?
    '''
    f = lambda x, t: (-3*x[0] + 3*x[0]*x[1], x[1] - 2*x[1]*x[0])
    x0 = [0.3, 1.0]
    odeint_sol = scipy.integrate.odeint(f, x0, T)
    #plot_graphs(X, Y, T, odeint_sol=odeint_sol)

    # question 1c.)
    '''
    (c) Find approximations for the minimum and maxi-mum number of predators
    that will be present.
    '''
    # according to euler
    max_x_euler, min_x_euler= int(round(max(X) * 1000)), int(round(min(X) * 1000))
    print('MAX Predetor = ', max_x_euler, ' @ approx ', 9, ' days')
    print('MIN Predetor = ', min_x_euler ,' @ approx ', 7, ' days')

    # question 1e.)
    '''
    (e) Suppose now that in the absense of predators, the limiting capacity of
    the environment for the prey is 10 (thousand). Add a suitable logistic term
    to the model above, solve the model using ode45 for a long enough time to
    estimate the behaviour of the two populations as t -> inf. Plot the result on a
    new figure.
    '''
    """
    NOTE: the folowing code does not converge accordingly, still needs some debuginh
    """
    t = np.arange(0, 1000, 5/10.0)
    f = lambda x, t: (-3*x[0] + 3*x[0]*x[1], x[1]-2*x[1]*x[0]-(1./10.)*x[1])
    x0 = [0.3, 1.0]
    odeint_sol = scipy.integrate.odeint(f, x0, t)

    plot_graph_Q1e(odeint_sol, t)

else:
    exit('USAGE: python problem1.py')
