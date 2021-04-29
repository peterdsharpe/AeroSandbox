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

my_node = Node(add, [lambda x: x, 2])

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
                 oper: Union[None, Callable] = None,
                 inputs: Union[None, List["Node"]] = None
                 ):
        if oper is None:
            oper = lambda x: x

        self.oper: Callable=oper
        self.inputs=inputs

    def __call__(self, x):
        if oper is None:
            if self.inputs is None:
                return x
            else:
                return


        if self.inputs==None:
            return x
        if input_size[self.oper]==1:
            return self.oper(self.inputs(x))
        else:
            return self.oper(self.inputs[0](x), self.inputs[1](x))


    def __str__(self):
        if self.inputs==None:
            return "No inputs"
        if input_size[self.oper]==1:
            return 'Operation: '+str(self.oper)+ '. Inputs: ' +str(self.inputs)
        return 'Operation: '+str(self.oper)+ '. Inputs: (' +str(self.inputs[0])+', '+str(self.inputs[1])+')'

class IndependentVariable():
    def __init__(self):
        pass

    def __call__(self, x):
        return x


# opti = asb.Opti()

def generate_trees(
        opers: List[Callable],
        size: int
):
    if size==0:
        yield Node(None, None)
    for i in opers:
        if input_size[i]==1:
            for k in generate_trees(opers, (size-1)):
                yield Node(i, k)
        if input_size[i]==2:
            for j in range((size-1)//2):
                for k in generate_trees(opers, j):
                    for l in generate_trees(opers, (size-1-j)):
                        yield Node(i, (k,l))
                        
            

    
if __name__ == '__main__':
    input_size={}
    input_size[add]=2

    my_node = Node(add, [Node(None, None), Node(None, None)])
    x = 3
    assert my_node(x) == 6

    my_node=Node(None, None)
    assert my_node(2)==2


    print(Node(add, [Node(add, None), Node(add, None)]).evaluate(1))