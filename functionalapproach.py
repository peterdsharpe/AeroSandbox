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
        return self.oper(self.inputs[0].evaluate(x), self.inputs[1].evaluate(x))
    
    def __str__(self):
        if self.inputs==None:
            return "Operation: "+str(self.oper)+". No inputs"
        return 'Operation: '+str(self.oper)+ '. Inputs: (' +str(self.inputs[0])+', '+str(self.inputs[1])+')'
        
opti = asb.Opti()   
def generate_trees(opers, size):
    yield opti.variable(init_guess=0)
    if size==0:
        return
    for i in opers:
        inputs=[]
        for j in i.input_size:
            for k in generate_trees(opers, (size-1)//i.input_size):
                inputs.append(k)
        yield Node(i, inputs)

    
if __name__ == '__main__':
    def add(x,y):
        return x+y
    print(Node(add, [Node(add, None), Node(add, None)]).evaluate(1))