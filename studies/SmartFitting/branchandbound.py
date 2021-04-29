#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:07:01 2021

@author: Sylvershadowz
"""
from data import x, y_data, f
import aerosandbox as asb
import aerosandbox.numpy as np

poly_degree=7
degree=poly_degree+3
eta=1

#vandermonde = np.ones((len(x), degree + 1))
#for j in range(1, degree + 1):
#    vandermonde[:, j] = vandermonde[:, j - 1] * x

def make_matrix(x):
    matrix = np.ones((len(x), degree + 1))
    for j in range(1, degree -2):
        matrix[:, j] = matrix[:, j - 1] * x
    matrix[:,degree-2]=np.cos(x)
    matrix[:,degree-1]=np.sin(x)
    matrix[:,degree]=np.exp(x)
    return matrix

matrix=make_matrix(x)

def model(coeffs):
    return matrix @ coeffs

def loss(y_model, y_data):
    return np.sum((y_model - y_data) ** 2)

#general branch and bound algorithm
def branch_and_bound(obj, lower_bound, branch, init, term, eta, guess=None):
    '''
    Uses branch and bound to minimize the objective
    obj- function to be minimized
    lower_bound- function that outputs a lower bound of an instance
    branch- function that produces two or more instances
    init- starting state with all variables uninitiated
    term- function that returns true if instance is a terminating condition
    eta- regularization factor
    guess- guess of an upper bound, can be produced by heuristic
    '''
    #sets the initial upper bound
    if guess!=None:
        best_value=guess(eta)
    else:
        best_value=float('inf')
    #creates a queue of instances to be checked
    q=[init]
    while q:
        n=q.pop()
        if term(n):
            if obj(n)[0]<=best_value:
                best_value=obj(n)[0]
                best_instance=n
                best_coeffs=obj(n)[1]
        
        else:
            n1, n2= branch(n)
            if lower_bound(n1)<=best_value:
                q.append(n1)
            if lower_bound(n2)<=best_value:
                q.append(n2)
    return best_instance, best_coeffs
    
def obj(n):
    '''
    minimizes objective (loss+regulaization) given n and eta
    n contains values (0 or 1) for each variable
    '''
    opti = asb.Opti()
    coeffs = opti.variable(init_guess=np.zeros(degree + 1))
    y_model=model(coeffs*n)
    error=loss(y_model, y_data)
    opti.minimize(
            error + eta*np.sum(n)
            )
    sol = opti.solve(verbose=False)
    
    return sol.value(error) + eta*np.sum(n), sol.value(coeffs)

def lower_bound(n):
    '''
    finds a lower bound for instance n
    '''
    
    opti = asb.Opti()
    coeffs = opti.variable(init_guess=np.zeros(degree + 1))
    n_new=np.where(n==None, 1, n)
    n_new=np.array(n_new)
    y_model=model(coeffs*n_new)
    error=loss(y_model, y_data)
    opti.minimize(
            error+eta*np.sum(np.where(n==None, 0, n))
            )
    sol = opti.solve(verbose=False)
    return sol.value(error)+eta*np.sum(np.where(n==None, 0, n))

def branch(n):
    n1=np.copy(n)
    n2=np.copy(n)
    for i in range(len(n)):
        if n[i]==None:
            n1[i]=0
            n2[i]=1
            return n1,n2
        
init=np.array([None]*(degree+1))

def term(n):
    return None not in n

def guess(eta):
    opti = asb.Opti()
    coeffs = opti.variable(init_guess=np.zeros(degree + 1))
    y_model=model(coeffs)
    error=loss(y_model, y_data)
    opti.minimize(
            error + eta*(degree + 1)
            )
    sol = opti.solve(verbose=False)
    
    return sol.value(error) + eta*(degree + 1)

if __name__ == '__main__':
    n, coeffs=branch_and_bound(obj, lower_bound, branch, init, term, eta, guess)
    print(n, coeffs)
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(palette=sns.color_palette("husl"))

    fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=200)

    x_plot = np.linspace(x[0], x[-1], 100)
#    vandermonde_plot = np.ones((len(x_plot), degree + 1))
#    for j in range(1, degree + 1):
#        vandermonde_plot[:, j] = vandermonde_plot[:, j - 1] * x_plot
    y_plot = make_matrix(x_plot) @ (coeffs*n).T
    
    x_extrapolate = np.linspace(x[0], x[-1]+10, 100)
    y_extrapolate=make_matrix(x_extrapolate) @ (coeffs*n).T

    plt.plot(x, y_data, ".")
    plt.plot(x_plot, y_plot, "-")
#    plt.plot(x_extrapolate, f(x_extrapolate))
#    plt.plot(x_extrapolate, y_extrapolate)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=200)
    plt.bar(
        x=np.arange(degree + 1),
        height=coeffs
    )
    plt.show()
