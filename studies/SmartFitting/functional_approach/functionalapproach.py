#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:02:28 2021

@author: Sylvershadowz
"""
import aerosandbox as asb
import aerosandbox.numpy as np
from typing import Callable, List, Tuple, Union
from functions import *

"""
hello

Define Node behavior:

Node(None, None) -> lambda x: x

# Node(None, 2) -> lambda x: 2

 

--------------------
# Paradigm 1

my_node = Node(add, [Node(None, None), Node(None, 2)])
my_node

x = 3

my_node.evaluate(x) = 6

--------------------
# Paradigm 1.5

my_node = Node(add, [IndependentVariable(), 2])

x = 3

my_node.evaluate(x) = 5

--------------------
# Paradigm 2

x = 3

my_node = Node(add, [x, 2])

my_node.evaluate() = 5

"""


class Node():
    def __init__(self,
                 oper: Callable,
                 inputs: List
                 ):
        '''
        Takes in an operation and a list
        List contains Nodes, IndependentVariable, constants, and opti variables
        '''

        self.oper: Callable=oper
        self.inputs=inputs

#    def __call__(self, x):
#        L=[]
#        for i in self.inputs:
#            if not callable(i):
#                L.append(i)
#            else:
#                L.append(i(x))
#        return self.oper(*L)


    def __str__(self):
        s= 'Operation: '+str(self.oper)+ '. Inputs: ('
        for i in self.inputs:
            s+= str(i)+ ', '
        s+=')'
        return s
    def get_sol(self, pnum=None):
        '''
        returns string representation of tree
        
        '''
        if pnum==None:
            pnum=[0]
        s=self.oper.__name__ +'('
        for i in self.inputs:
            if isinstance(i, Node):
                s+= i.get_sol(pnum)+ ', '
            elif isinstance(i, IndependentVariable):
                s+= str(i)+ ', '
            else:
                pnum[0]+=1
                s+= i+str(pnum[0])+ ', '
        self.pnum=pnum[0]
        return s[:-2]+')'
            
    def __call__(self, x, params):
        '''
        Substitutes params as optimization parameters and evaluates tree with
        independent variable x
        '''
        s=self.get_sol()
        for i in range(len(params)):
            exec('p'+str(i+1)+'=params['+str(i)+']')
        return eval(s)
    
    def optimize(self, x_data, y_data, loss):
        '''
        Finds optimal parameters given data and loss
        '''
        
        opti = asb.Opti()
        string_sol=self.get_sol()
        L=[]
        x=x_data
        for j in range(self.pnum):
            exec('p'+str(j+1)+'='+'opti.variable(init_guess=1)')
            exec('L.append(p'+str(j+1)+')')
        predicted=eval(string_sol)
        error=loss(predicted, y_data)
        opti.minimize(
                error
                )
        try:
            sol = opti.solve(max_iter=500, verbose=False)
        except RuntimeError:
            sol = opti.debug
        return sol.value(error), sol, L
            
class IndependentVariable():
    def __init__(self):
        pass

    def __call__(self, x):
        return x
    
    def __str__(self):
        return 'x'




def generate_trees(
        opers: List[Callable],
        size: int,
):
    if size<=0:
        yield IndependentVariable()
        yield 'p'
        return
    for i in opers:
        if input_size[i]==1:
            for k in generate_trees(opers, (size-1)):
                yield Node(i, (k,))
        if input_size[i]==2:
            for j in range(size):
                for k in generate_trees(opers, j):
                    for l in generate_trees(opers, (size-1-j)):
                        yield Node(i, (k,l))
          
def best_tree_of_size(x_data, y_data, loss, opers, size):
    '''
    Generates the best tree given a size by looping through all possibilities
    '''
    bestvalue=float('inf')
    g=generate_trees(opers, size)
    for i in g:
        errorvalue, sol, L=i.optimize(x_data, y_data, loss)
        if errorvalue<bestvalue:
            bestvalue=errorvalue
            bestsol=sol
            besttree=i
            bestparams=L
        
    paramvalues=[]
    for i in bestparams:
        paramvalues.append(bestsol.value(i))
    return besttree, paramvalues, bestvalue

def best_tree_dynamic(x_data, y_data, loss, opers, size):
    if size<=1:
        return best_tree_of_size(x_data, y_data, loss, opers, size)
    prevtree, prevvalues, preverror=best_tree_dynamic(x_data, y_data, loss, opers, size-1)
    bestvalue=float('inf')
    for i in extended_trees(prevtree, opers):
        errorvalue, sol, L=i.optimize(x_data, y_data, loss)
        if errorvalue<bestvalue:
            bestvalue=errorvalue
            bestsol=sol
            besttree=i
            bestparams=L
    paramvalues=[]
    for i in bestparams:
        paramvalues.append(bestsol.value(i))
    return besttree, paramvalues, bestvalue
    
def extended_trees(tree, opers):
    '''
    Generates all possible trees made by extending the current tree by 1 node
    '''
    #base case
    if isinstance(tree, str) or isinstance(tree, IndependentVariable):
        for j in generate_trees(opers, 1):
            yield j
        return            
    #adds an additional node at the root
    for i in opers:
        if input_size[i]==1:
            yield Node(i, (tree,))
        else:
            yield Node(i, (tree, 'p'))
            yield Node(i, (tree, IndependentVariable()))
            yield Node(i, ('p', tree))
            yield Node(i, (IndependentVariable(), tree))
    #recursive case
    for i in range(len(tree.inputs)):
        for j in extended_trees(tree.inputs[i], opers):
            yield Node(tree.oper, tree.inputs[:i]+(j,)+tree.inputs[i+1:])
            
    
    
    
    
            

    
if __name__ == '__main__':
    input_size={}
    input_size[add]=2
    input_size[square]=1
    input_size[multiply]=2
    input_size[sin]=1
    input_size[ln]=1
    
    
#    g=generate_trees([add, square], 2)
#    for i in g:
#        print('tree:', i)
#        x=3
#        print(i.get_sol())
#        print('\n')
    
    
    
    
        
    np.random.seed(0)
    
    
    

    x = np.linspace(1, 10, 20)
    
    def f(x):
        return 10+x**2+10*np.sin(x)
    def loss(y_model, y_data):
        return np.sum((y_model - y_data) ** 2)
    
    
    
    y_data = f(x)  + 0.1 * np.random.randn(len(x))
    
    
    
#    opti = asb.Opti()
#    p1=opti.variable(init_guess=1)
#    p2=opti.variable(init_guess=1)
#    p3=opti.variable(init_guess=1)
#    p4=opti.variable(init_guess=1)
#    predicted=add(multiply(add(x, p1), add(add(x, multiply(sin(x), p2)), p3)), p4)
#
#    error=loss(predicted, y_data)
#    opti.minimize(
#            error + 1e-9 * (p1 ** 2 + p2 ** 2 + p3 **2 + p4 **2)
#            )
#    # sol=opti.solve(verbose=True)
#    try:
#        sol = opti.solve(max_iter=500)
#    except RuntimeError:
#        sol = opti.debug
#    
#    for var in [p1, p2, p3, p4]:
#        print(sol.value(var))
#    
#    raise Exception()
    
    
    
#    besttree, paramvalues, errorvalue=best_tree_of_size(x, y_data, loss, [add, square], 3)
#    print(besttree.get_sol())
#    index=1
#    print(paramvalues)
#    print(errorvalue)
    
    besttree, paramvalues, errorvalue=best_tree_dynamic(x, y_data, loss, [add, square, multiply, sin], 10)
    print(besttree.get_sol())
    print(paramvalues)
    
    for t in extended_trees(Node(add, ('p','p')), [add, square]):
        print(t.get_sol())
        
    
    

    
    ### Expression object
    # __repr__ returns something like: "sum(sum(times(p1, square(x)), times(p2, x)), p3)"
    #           or, simply something like "p1*x**2+p2*x+p3"
    #  Expression has no dependency on the data, or on an Opti environment
    # def substitute(self, p): Given some vector p (either scalars, or a vector of optimization
            # variables), return a version of Expression with the p values substituted in.
    # alternative to substitute: define __call__(self, x, p): This way, it's compatible with asb.FittedModel
    
    #add(x, add(p1, p2))
    
    #dynamic program- start with depth 1, use as guess for depth 2, etc, randomaly throw is mutation
    #multidimensional data
    
    #branch at intermediate nodes
    
    #do excpet to catch runtime
    #check for other initial guesses