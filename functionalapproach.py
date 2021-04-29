#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:02:28 2021

@author: Sylvershadowz
"""
import aerosandbox as asb
import aerosandbox.numpy as np

class Node():
    def __init__(self, oper, inputs):
        self.oper=oper
        self.inputs=inputs
        
    
    def evaluate(self, x):
        if self.inputs==None:
            return x
        if input_size[self.oper]==1:
            return self.oper(self.inputs.evaluate(x))
        else:
            return self.oper(self.inputs[0].evaluate(x), self.inputs[1].evaluate(x))
        
    
    def __str__(self):
        if self.inputs==None:
            return "No inputs"
        if input_size[self.oper]==1:
            return 'Operation: '+str(self.oper)+ '. Inputs: ' +str(self.inputs)
        return 'Operation: '+str(self.oper)+ '. Inputs: (' +str(self.inputs[0])+', '+str(self.inputs[1])+')'
        
opti = asb.Opti()   
def generate_trees(opers, size):
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
    def add(x,y):
        return x+y
    print(Node(add, [Node(add, None), Node(add, None)]).evaluate(1))