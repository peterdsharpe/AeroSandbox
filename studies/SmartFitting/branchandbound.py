#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 18:07:01 2021

@author: Sylvershadowz
"""
from data import x, y_data
import aerosandbox as asb
import aerosandbox.numpy as np

degree=2

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
            if obj(n)<best_value:
                best_value=obj(n)
                best_instance=n
        n1, n2= branch(n)
        if lower_bound(n1)<=best_value:
            q.append(n1)
        if lower_bound(n2)<=best_value:
            q.append(n2)
    return best_instance
    
def obj(n, eta):
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
    
    return error + eta*np.sum(n)

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
    
    return error

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
    if None not in n:
        return True
    return False

def guess(eta):
    opti = asb.Opti()
    coeffs = opti.variable(init_guess=np.zeros(degree + 1))
    y_model=model(coeffs)
    error=loss(y_model, y_data)
    opti.minimize(
            error + eta*(degree + 1)
            )
    sol = opti.solve(verbose=False)
    
    return error + eta*(degree + 1)
