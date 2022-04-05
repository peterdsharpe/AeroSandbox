#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:36:45 2021

@author: Sylvershadowz
"""
import random
def simann(func, temp, neighbor, prob, init, steps):
    state=init
    for i in range(steps):
        t=temp((i+1)/steps)
        newstate=neighbor(state)
        if prob(func(state), func(newstate), t) <random.random():
            state=newstate
        return state

